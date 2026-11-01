from abc import ABC, abstractmethod
from typing import Optional

from core.workspace.proposal import Proposal

# A need competes for the workspace once it exceeds this level.
NEED_THRESHOLD = 0.3
# Matches DecisionMaker's critical-need override (> 0.85): such a need pre-empts everything else.
CRITICAL_LEVEL = 0.85
CRITICAL_BONUS = 0.5
# Experimental switch: when True, critical needs outrank each other in DecisionMaker's fixed order (hunger,
# then thirst, then fatigue) instead of by how high each level is. Off by default.
FIXED_NEED_PRIORITY = False
_FIXED_RANK = {"hunger": 2.0, "thirst": 1.0, "fatigue": 0.0}
# Goals compete at priority * GOAL_WEIGHT, so a need above ~sqrt(0.45) = 0.67 outbids a priority-0.9 goal.
GOAL_WEIGHT = 0.5
# The learned DQN policy is the fallback when nothing more specific bids higher.
HABIT_SALIENCE = 0.3

# The beacon cue competes for a workspace slot at this salience (it never acts by itself).
BEACON_SALIENCE = 0.6
# Experimental control: report the wrong half. The cue then carries content, but misleading content.
BEACON_INVERT = False
# How the beacon bids: 'constant' (every step at BEACON_SALIENCE) or 'interval' (silent until the cue has been out
# of the workspace for BEACON_INTERVAL steps, then a bid of BEACON_FORCE, high enough to win the slot) or 'oracle'
# (bids BEACON_FORCE exactly when the remembered cue differs from the true active half: it cheats, as an upper bound
# on what any attention policy could do).
BEACON_POLICY = "constant"
BEACON_INTERVAL = 25
BEACON_FORCE = 10.0
NEVER = 10 ** 9

# need -> (resource it is satisfied by, action that consumes/searches for it)
_NEEDS = {
    "hunger": ("food", "seek_food"),
    "thirst": ("water", "drink_water"),
    "fatigue": (None, "rest"),
}


def perceived_here(agent) -> Optional[str]:
    """What the agent perceives on its own cell (centre of the 3x3 local view), if anything."""
    view = agent.perception.get("local_grid_view") or []
    if len(view) == 3 and len(view[1]) == 3:
        return view[1][1]
    return None


class Specialist(ABC):
    """
    One parallel, specialised processor. A specialist reads the agent, never another
    specialist, and communicates only by bidding for the global workspace.
    """
    name: str

    @abstractmethod
    def propose(self, agent) -> Optional[Proposal]:
        """Returns this specialist's bid for the current step, or None to stay silent."""


def read_regime(agent) -> Optional[str]:
    """The beacon sensor: which half of the world holds resources now (None without the regime)."""
    half = getattr(agent.environment, "active_half", None)
    if half is not None and BEACON_INVERT:
        return "right" if half == "left" else "left"
    return half


class BeaconSpecialist(Specialist):
    """The only reader of the beacon sensor. Its cue reaches other modules only if it wins a workspace slot."""

    name = "beacon"

    def propose(self, agent) -> Optional[Proposal]:
        cue = read_regime(agent)
        if cue is None:
            return None
        if BEACON_POLICY == "oracle":
            if getattr(agent, "cue_trace", None) == cue:
                return None
            return Proposal(self.name, "none", BEACON_FORCE, {"cue": cue})
        if BEACON_POLICY == "interval":
            if getattr(agent, "cue_age", NEVER) < BEACON_INTERVAL:
                return None
            return Proposal(self.name, "none", BEACON_FORCE, {"cue": cue})
        return Proposal(self.name, "none", BEACON_SALIENCE, {"cue": cue})


class NeedSpecialist(Specialist):
    """Bids to satisfy one physiological need once it is pressing."""

    def __init__(self, need: str):
        self.need = need
        self.name = f"need:{need}"
        self.resource, self.action = _NEEDS[need]

    def propose(self, agent) -> Optional[Proposal]:
        level = getattr(agent.internal_state, self.need)
        if level < NEED_THRESHOLD:
            return None
        salience = level ** 2 + (CRITICAL_BONUS if level > CRITICAL_LEVEL else 0.0)
        if FIXED_NEED_PRIORITY and level > CRITICAL_LEVEL:
            salience += _FIXED_RANK[self.need]
        consume_here = self.resource is not None and perceived_here(agent) == self.resource
        return Proposal(self.name, self.action, salience,
                        {"need": self.need, "level": level, "consume_here": consume_here})


class GoalSpecialist(Specialist):
    """Bids to take the next step toward the highest-priority unfinished location goal."""

    name = "goal"

    def propose(self, agent) -> Optional[Proposal]:
        goals = [g for g in agent.active_goals if g["type"] == "reach_location" and not g["completed"]]
        if not goals:
            return None
        goal = max(goals, key=lambda g: g["priority"])
        target = (goal["target_x"], goal["target_y"])
        if (agent.pos_x, agent.pos_y) == target:
            goal["completed"] = True
            return None
        return Proposal(self.name, self._step_toward(agent, target), goal["priority"] * GOAL_WEIGHT,
                        {"goal": goal["id"], "target": target})

    @staticmethod
    def _step_toward(agent, target) -> str:
        """The move that closes the larger gap first; push an obstacle if every useful move is blocked."""
        d_row, d_col = target[0] - agent.pos_x, target[1] - agent.pos_y
        candidates = []
        if d_row:
            candidates.append((abs(d_row), "move_down" if d_row > 0 else "move_up"))
        if d_col:
            candidates.append((abs(d_col), "move_right" if d_col > 0 else "move_left"))
        view = agent.perception.get("local_grid_view") or []
        for _, move in sorted(candidates, reverse=True):
            row_off, col_off = agent._MOVE_OFFSETS[move]
            blocked = len(view) == 3 and len(view[1 + row_off]) == 3 and view[1 + row_off][1 + col_off] == 'obstacle'
            if not blocked:
                return move
        return "move_object"


class HabitSpecialist(Specialist):
    """Bids with the learned DQN policy at a low, constant salience."""

    name = "habit"

    def propose(self, agent) -> Optional[Proposal]:
        state = agent.internal_state
        action = agent.q_learner.choose_action(hunger=state.hunger, fatigue=state.fatigue, thirst=state.thirst)
        return Proposal(self.name, action, HABIT_SALIENCE, {"policy": "dqn"})
