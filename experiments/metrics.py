"""Metric collection and aggregation for headless runs."""

import io
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

# Level at which DecisionMaker treats a need as critical (hunger/thirst/fatigue > 0.85).
CRITICAL_LEVEL = 0.85

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
        self.steps = 0

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
        result["action_counts"] = dict(self.actions)
        result["consciousness_steps"] = dict(self.states)
        return result


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarises per-agent metrics pooled over every agent of every run."""
    samples = [metrics for run in runs for metrics in run["agents"].values()]
    summary: Dict[str, Any] = {"agent_runs": len(samples)}

    scalar_keys = [k for k, v in samples[0].items()
                   if isinstance(v, (int, float)) and k != "reach_center_step"]
    for key in scalar_keys:
        values = [s[key] for s in samples]
        summary[key] = {"mean": mean(values), "std": pstdev(values),
                        "min": min(values), "max": max(values)}

    completed = [s["reach_center_step"] for s in samples if s["reach_center_step"] is not None]
    summary["reach_center"] = {
        "completed_rate": len(completed) / len(samples),
        "mean_step": mean(completed) if completed else None,
    }

    for key in ("action_counts", "consciousness_steps"):
        names = sorted({name for s in samples for name in s[key]})
        summary[key] = {name: mean(s[key].get(name, 0) for s in samples) for name in names}
    return summary
