from collections import Counter
from typing import List, Optional, Sequence

from core.workspace.proposal import Proposal
from core.workspace.specialists import (
    GoalSpecialist, HabitSpecialist, NeedSpecialist, Specialist)

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
    """

    def __init__(self, specialists: Sequence[Specialist], capacity: int = 1,
                 persistence: float = 0.15, feedback: bool = True):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.specialists = list(specialists)
        self.capacity = capacity
        self.persistence = persistence
        self.feedback = feedback
        self.contents: List[Proposal] = []
        self.winner_counts: Counter = Counter()
        self.opportunistic_actions = 0
        self._last_winner_key = None

    @classmethod
    def default(cls, capacity: int = 1, persistence: float = 0.15, feedback: bool = True) -> "GlobalWorkspace":
        """The standard specialist set: one per need, one for goals, and the learned habit."""
        specialists = [NeedSpecialist("hunger"), NeedSpecialist("thirst"), NeedSpecialist("fatigue"),
                       GoalSpecialist(), HabitSpecialist()]
        return cls(specialists, capacity, persistence, feedback)

    def decide(self, agent) -> str:
        """Runs one competition and returns the action chosen from the broadcast content."""
        proposals = [p for p in (s.propose(agent) for s in self.specialists) if p is not None]

        def effective_salience(proposal: Proposal) -> float:
            bonus = self.persistence if proposal.key == self._last_winner_key else 0.0
            return proposal.salience + bonus

        self.contents = sorted(proposals, key=effective_salience, reverse=True)[:self.capacity]
        winner = self.contents[0]
        self._last_winner_key = winner.key
        self.winner_counts[winner.source] += 1

        action = self._select_action(winner)
        if self.feedback:
            self._steer_attention(agent, winner)

        agent.internal_monologue += (
            f"Global workspace broadcast {[f'{p.source}:{p.action}' for p in self.contents]}; "
            f"winner '{winner.source}'. ")
        return action

    def _select_action(self, winner: Proposal) -> str:
        """The winner's action, unless another admitted need can be satisfied right here, right now."""
        if not winner.content.get("consume_here"):
            for item in self.contents[1:]:
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
