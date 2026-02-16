"""Q-Table reinforcement learner with ε-greedy exploration."""

import random
from collections import defaultdict


class QTableLearner:
    """Tabular Q-learning agent over discretized (hunger, fatigue) state space.

    State is discretized into 11×11 bins. Action selection uses
    ε-greedy: random with probability ε, greedy otherwise.

    Args:
        actions: List of valid action strings.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration probability.
    """

    def __init__(
        self,
        actions: list[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: defaultdict = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )

    def get_state_key(self, hunger: float, fatigue: float) -> str:
        """Discretize continuous state into a string key."""
        h = int(hunger * 10)
        f = int(fatigue * 10)
        return f"h{h}_f{f}"

    def choose_action(self, hunger: float, fatigue: float) -> str:
        """Select action via ε-greedy policy."""
        state = self.get_state_key(hunger, fatigue)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(
        self,
        prev_hunger: float,
        prev_fatigue: float,
        action: str,
        reward: float,
        next_hunger: float,
        next_fatigue: float,
    ) -> None:
        """Update Q-value for (state, action) using Bellman equation."""
        prev_state = self.get_state_key(prev_hunger, prev_fatigue)
        next_state = self.get_state_key(next_hunger, next_fatigue)

        max_future_q = max(self.q_table[next_state].values())
        old_q = self.q_table[prev_state][action]

        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[prev_state][action] = new_q

    def __str__(self) -> str:
        sample = dict(list(self.q_table.items())[:5])
        return f"Q-table ({len(self.q_table)} states, sample: {sample})"
