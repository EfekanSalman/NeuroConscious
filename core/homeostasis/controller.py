from collections import Counter
from typing import List

from core.homeostasis.drive import drive_of_state
from core.learning.dqn_learner import DQNLearner

# The high-level options the controller chooses between. 'seek_food' and 'drink_water' are search
# actions (they consume the resource when standing on it, otherwise step toward it).
OPTIONS: List[str] = ["seek_food", "drink_water", "rest"]


class HomeostaticController:
    """
    Chooses which need to attend to by reinforcement learning on homeostatic drive reduction.

    The reward is the one of Keramati & Gutkin (2014): r = D(H_t) - D(H_t+1), the reduction of
    the drive that the agent's deficits jointly define. No behaviour is prescribed. Because the
    drive is convex, the learned policy can come to act before a need becomes critical.

    Args:
        learner (DQNLearner): Q-network over OPTIONS with the state (hunger, fatigue, thirst).
        reward_n (float): Deviation exponent of the drive used as reward.
        reward_m (float): Root exponent of the drive used as reward.
    """

    def __init__(self, learner: DQNLearner, reward_n: float = 4, reward_m: float = 2):
        self.learner = learner
        self.reward_n = reward_n
        self.reward_m = reward_m
        self.option_counts: Counter = Counter()
        self.total_reward = 0.0

    @classmethod
    def default(cls, gamma: float = 0.95, learning_rate: float = 0.001, epsilon_decay: float = 0.999,
                epsilon_min: float = 0.05, reward_n: float = 4, reward_m: float = 2) -> "HomeostaticController":
        learner = DQNLearner(actions=OPTIONS, state_size=3, learning_rate=learning_rate, gamma=gamma,
                             epsilon_decay_rate=epsilon_decay, epsilon_min=epsilon_min, batch_size=32)
        return cls(learner, reward_n, reward_m)

    def decide(self, agent) -> str:
        """The option to pursue now, chosen epsilon-greedily from the learned values."""
        state = agent.internal_state
        option = self.learner.choose_action(hunger=state.hunger, fatigue=state.fatigue, thirst=state.thirst)
        self.option_counts[option] += 1
        return option

    def reward(self, previous, current) -> float:
        """Drive reduction between two states (positive when the agent got closer to its set points)."""
        return (drive_of_state(previous, self.reward_n, self.reward_m)
                - drive_of_state(current, self.reward_n, self.reward_m))

    def learn(self, previous: dict, option: str, current) -> float:
        """Learns from one step: `previous` is the state snapshot before acting, `current` the state after."""
        reward = self.reward(previous, current)
        self.total_reward += reward
        self.learner.update(previous["hunger"], previous["fatigue"], previous["thirst"], option, reward,
                            current.hunger, current.fatigue, current.thirst)
        return reward
