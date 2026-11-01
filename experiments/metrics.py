"""Metric collection and aggregation for headless runs."""

import io
import re
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

from core.homeostasis import drive_of_state

# Level at which DecisionMaker treats a need as critical (hunger/thirst/fatigue > 0.85).
CRITICAL_LEVEL = 0.85
# Drive is also reported per block of this many steps, to show change over a long run (learning).
DRIVE_BLOCK = 500

# The simulation reports discrete events only through print(). These patterns match those
# messages (agent/base_agent.py::_act_default), so they must change together with them.
_EVENT_PATTERNS = {
    "meals": re.compile(r"^(\S+) successfully ate food"),
    "drinks": re.compile(r"^(\S+) successfully drank water"),
    "pushes": re.compile(r"^(\S+) successfully moved object"),
    "failed_pushes": re.compile(r"^(\S+) tried to move an object"),
    "obstacle_bumps": re.compile(r"^(\S+) tried to move \w+ onto an obstacle"),
    "out_of_bounds": re.compile(r"^(\S+) tried to move \w+ out of bounds"),
}


class EventCounter(io.TextIOBase):
    """Write-only stream that tallies simulation events per agent and discards the text."""

    def __init__(self):
        self.counts: Dict[str, Counter] = defaultdict(Counter)
        self._partial = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            for event, pattern in _EVENT_PATTERNS.items():
                match = pattern.match(line)
                if match:
                    self.counts[match.group(1)][event] += 1
                    break
        return len(text)


class AgentTracker:
    """Accumulates per-step observations of one agent; call observe() after each world step."""

    def __init__(self, agent):
        self.agent = agent
        self.cells = set()
        self.last_pos = None
        self.moves = 0
        self.stationary_steps = 0
        self.level_sums = {"hunger": 0.0, "thirst": 0.0, "fatigue": 0.0}
        self.critical_steps = {"hunger": 0, "thirst": 0, "fatigue": 0}
        self.actions: Counter = Counter()
        self.states: Counter = Counter()
        self.reach_center_step: Optional[int] = None
        self.drive_sum = 0.0
        self.drive_blocks: List[float] = []
        self.steps = 0
        self.cue_steps = 0
        self.cue_correct_steps = 0
        self.held_cue_steps = 0  # Steps with a usable cue: in the workspace or remembered.
        self.held_cue_correct_steps = 0
        self.pressure_steps = {"calm": 0, "urgent": 0}  # Steps by the highest need level: below 0.6 / above 0.85.
        self.pressure_cue_steps = {"calm": 0, "urgent": 0}

    def observe(self, step: int) -> None:
        agent = self.agent
        pos = (agent.pos_x, agent.pos_y)
        self.cells.add(pos)
        if self.last_pos is not None:
            if pos != self.last_pos:
                self.moves += 1
            else:
                self.stationary_steps += 1
        self.last_pos = pos

        for need in self.level_sums:
            level = getattr(agent.internal_state, need)
            self.level_sums[need] += level
            if level > CRITICAL_LEVEL:
                self.critical_steps[need] += 1

        level = drive_of_state(agent.internal_state)
        self.drive_sum += level
        if self.steps % DRIVE_BLOCK == 0:
            self.drive_blocks.append(0.0)
        self.drive_blocks[-1] += level

        half = getattr(agent.environment, "active_half", None)
        if half is not None and agent.regime_cue is not None:
            self.cue_steps += 1
            self.cue_correct_steps += agent.regime_cue == half
        held = agent.effective_cue() if half is not None else None
        if held is not None:
            self.held_cue_steps += 1
            self.held_cue_correct_steps += held == half
        if half is not None:
            top = max(agent.internal_state.hunger, agent.internal_state.thirst, agent.internal_state.fatigue)
            band = "calm" if top < 0.6 else "urgent" if top > CRITICAL_LEVEL else None
            if band:
                self.pressure_steps[band] += 1
                self.pressure_cue_steps[band] += agent.regime_cue is not None
        self.actions[agent._last_performed_action] += 1
        self.states[agent.current_consciousness_state.get_state_name()] += 1

        if self.reach_center_step is None and any(
                g["id"] == "goal_reach_center" and g["completed"] for g in agent.active_goals):
            self.reach_center_step = step
        self.steps += 1

    def result(self, events: Counter, grid_size: int) -> Dict[str, Any]:
        steps = max(self.steps, 1)
        state = self.agent.internal_state
        result: Dict[str, Any] = {event: events.get(event, 0) for event in _EVENT_PATTERNS}
        result.update({
            "moves": self.moves,
            "stationary_steps": self.stationary_steps,
            "distinct_cells": len(self.cells),
            "coverage": len(self.cells) / (grid_size * grid_size),
        })
        for need, total in self.level_sums.items():
            result[f"{need}_mean"] = total / steps
            result[f"{need}_critical_steps"] = self.critical_steps[need]
            result[f"{need}_final"] = getattr(state, need)
        result["reach_center_step"] = self.reach_center_step
        blocks = [total / min(DRIVE_BLOCK, self.steps - i * DRIVE_BLOCK) for i, total in enumerate(self.drive_blocks)]
        result["drive_mean"] = self.drive_sum / steps
        # Anticipation: the need level at which the agent eats / drinks (None if it never did).
        for need, name in (("hunger", "hunger_at_meal_median"), ("thirst", "thirst_at_drink_median")):
            levels = self.agent.consumption_levels[need]
            result[name] = median(levels) if levels else None
        result["drive_last_block"] = blocks[-1] if blocks else 0.0
        result["drive_by_block"] = blocks
        result["action_counts"] = dict(self.actions)
        result["consciousness_steps"] = dict(self.states)
        if getattr(self.agent.environment, "active_half", None) is not None:
            result["cue_steps"] = self.cue_steps
            result["cue_correct_steps"] = self.cue_correct_steps
            result["held_cue_steps"] = self.held_cue_steps
            result["held_cue_correct_steps"] = self.held_cue_correct_steps
            for band in ("calm", "urgent"):
                result[f"{band}_steps"] = self.pressure_steps[band]
                result[f"{band}_cue_steps"] = self.pressure_cue_steps[band]
        if getattr(self.agent, "attention_stats", None) and self.agent.attention_stats["intended_steps"]:
            result.update({f"attention_{k}": v for k, v in self.agent.attention_stats.items()})
        workspace = getattr(self.agent, "workspace", None)
        if workspace is not None:
            result["opportunistic_actions"] = workspace.opportunistic_actions
            result["workspace_winners"] = dict(workspace.winner_counts)
        return result


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarises per-agent metrics pooled over every agent of every run."""
    samples = [metrics for run in runs for metrics in run["agents"].values()]
    summary: Dict[str, Any] = {"agent_runs": len(samples)}

    scalar_keys = sorted({k for s in samples for k, v in s.items()
                          if isinstance(v, (int, float)) and k != "reach_center_step"})
    for key in scalar_keys:
        values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        summary[key] = {"mean": mean(values), "std": pstdev(values),
                        "min": min(values), "max": max(values)}

    completed = [s["reach_center_step"] for s in samples if s["reach_center_step"] is not None]
    summary["reach_center"] = {
        "completed_rate": len(completed) / len(samples),
        "mean_step": mean(completed) if completed else None,
    }

    blocks = [s["drive_by_block"] for s in samples if "drive_by_block" in s]
    if blocks:
        longest = max(len(b) for b in blocks)
        summary["drive_by_block"] = [mean(b[i] for b in blocks if len(b) > i) for i in range(longest)]

    for key in ("action_counts", "consciousness_steps", "workspace_winners"):
        if key not in samples[0]:
            continue
        names = sorted({name for s in samples for name in s[key]})
        summary[key] = {name: mean(s[key].get(name, 0) for s in samples) for name in names}
    return summary
