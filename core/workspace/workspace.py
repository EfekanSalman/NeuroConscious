from collections import Counter
from typing import List, Optional, Sequence

from core.workspace.memory import WorkspaceMemory
from core.workspace.proposal import Proposal
from core.workspace.specialists import (
    BeaconSpecialist, GoalSpecialist, HabitSpecialist, NeedSpecialist, Specialist)

# Experimental switch for ablations: names of specialists left out of the default set ('goal', 'habit').
DISABLED_SPECIALISTS: tuple = ()

# Which input channel the agent's perception should favour when a given need wins the workspace.
_ATTENTION_FOR_NEED = {"hunger": "food", "thirst": "water", "fatigue": "rest"}


class GlobalWorkspace:
    """
    A limited-capacity global workspace in the sense of Global Workspace Theory.

    Each step every specialist bids in parallel (GWT-1). Only the `capacity` most salient bids
    are admitted (GWT-2). The admitted content is broadcast: the action is chosen from it alone,
    and it feeds back to the input side by steering perceptual attention (GWT-3). The competition
    is state-dependent: a bid on the same topic as last step's winner gets a persistence bonus,
    so what is already in the workspace biases what enters next (GWT-4).

    Args:
        specialists (Sequence[Specialist]): The parallel bidders.
        capacity (int): How many bids are admitted and broadcast per step.
        persistence (float): Salience bonus for the topic that won the previous step.
        feedback (bool): Whether the broadcast steers the agent's perceptual attention.
        memory (WorkspaceMemory, optional): Sustained content that new percepts compete to enter.
            When set, the agent recalls resource locations from it instead of its FIFO buffer.
    """

    def __init__(self, specialists: Sequence[Specialist], capacity: int = 1,
                 persistence: float = 0.15, feedback: bool = True,
                 memory: Optional[WorkspaceMemory] = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.specialists = list(specialists)
        self.capacity = capacity
        self.persistence = persistence
        self.feedback = feedback
        self.memory = memory
        self.contents: List[Proposal] = []
        self.winner_counts: Counter = Counter()
        self.opportunistic_actions = 0
        self._last_winner_key = None

    @classmethod
    def default(cls, capacity: int = 1, persistence: float = 0.15, feedback: bool = True,
                memory_slots: int = 0, memory_policy: str = "salience",
                memory_recall: str = "nearest", beacon: bool = False) -> "GlobalWorkspace":
        """The standard specialist set: one per need, one for goals, and the learned habit.

        memory_slots > 0 adds a WorkspaceMemory of that many sightings with the given policy."""
        specialists = [NeedSpecialist("hunger"), NeedSpecialist("thirst"), NeedSpecialist("fatigue"),
                       GoalSpecialist(), HabitSpecialist()]
        specialists = [s for s in specialists if s.name not in DISABLED_SPECIALISTS]
        if beacon:
            specialists.append(BeaconSpecialist())
        memory = WorkspaceMemory(memory_slots, memory_policy, memory_recall) if memory_slots > 0 else None
        return cls(specialists, capacity, persistence, feedback, memory)

    def decide(self, agent) -> str:
        """Runs one competition and returns the action chosen from the broadcast content."""
        if self.memory is not None:
            self.memory.observe(agent)
        proposals = [p for p in (s.propose(agent) for s in self.specialists) if p is not None]
        acting = [p for p in proposals if "cue" not in p.content]
        if not acting:  # Nothing is pressing and no habit fills in: the agent idles.
            self.contents = []
            self._broadcast_cue(agent)
            self.winner_counts["none"] += 1
            agent.internal_monologue += "Global workspace: nothing bid, idling. "
            return "explore"

        def effective_salience(proposal: Proposal) -> float:
            bonus = self.persistence if proposal.key == self._last_winner_key else 0.0
            return proposal.salience + bonus

        self.contents = sorted(proposals, key=effective_salience, reverse=True)[:self.capacity]
        if not any("cue" not in p.content for p in self.contents):  # A cue never wins the action by itself.
            self.contents = [max(acting, key=effective_salience)]
        winner = next(p for p in self.contents if "cue" not in p.content)
        self._broadcast_cue(agent)
        self._last_winner_key = winner.key
        self.winner_counts[winner.source] += 1

        action = self._select_action(winner)
        if self.feedback:
            self._steer_attention(agent, winner)

        agent.internal_monologue += (
            f"Global workspace broadcast {[f'{p.source}:{p.action}' for p in self.contents]}; "
            f"winner '{winner.source}'. ")
        return action

    def _broadcast_cue(self, agent) -> None:
        """The cue is available to the rest of the agent only on steps when it holds a workspace slot."""
        if any(s.name == "beacon" for s in self.specialists):
            cues = [p.content["cue"] for p in self.contents if "cue" in p.content]
            agent.regime_cue = cues[0] if cues else None

    def _select_action(self, winner: Proposal) -> str:
        """The winner's action, unless another admitted need can be satisfied right here, right now."""
        if not winner.content.get("consume_here"):
            for item in (p for p in self.contents if p is not winner):
                if item.content.get("consume_here"):
                    self.opportunistic_actions += 1
                    return item.action
        return winner.action

    @staticmethod
    def _steer_attention(agent, winner: Proposal) -> None:
        """Broadcast reaches the input side: perception favours what the winning content needs."""
        need = winner.content.get("need")
        if need:
            agent.attention_focus = _ATTENTION_FOR_NEED[need]
        elif winner.content.get("goal"):
            agent.attention_focus = "location_target"
