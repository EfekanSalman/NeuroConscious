import random
from collections import defaultdict

class QTableLearner:
    """
    Implements a Q-learning agent that learns optimal actions based on states and rewards.

    Q-learning is a model-free reinforcement learning algorithm to learn a policy
    telling an agent what action to take under what circumstances. It uses a Q-table
    to store the 'quality' (expected utility) of taking a given action in a given state.
    """
    def __init__(self, actions: list[str], alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        """
        Initializes the QTableLearner.

        Args:
            actions (list[str]): A list of possible actions the agent can take (e.g., ["seek_food", "rest"]).
            alpha (float, optional): The learning rate (alpha). Controls how much new information
                                     overrides old information. Defaults to 0.1.
            gamma (float, optional): The discount factor (gamma). Determines the importance of
                                     future rewards. Defaults to 0.9.
            epsilon (float, optional): The exploration rate (epsilon). The probability of
                                       the agent choosing a random action (exploration) instead of
                                       the best known action (exploitation). Defaults to 0.2.
        """
        # The Q-table stores Q-values for state-action pairs.
        # defaultdict is used so that new states are initialized with all actions having Q-value 0.0.
        self.q_table: defaultdict[str, dict[str, float]] = defaultdict(lambda: {action: 0.0 for action in actions})
        self.alpha: float = alpha       # Learning rate
        self.gamma: float = gamma       # Discount factor for future rewards
        self.epsilon: float = epsilon   # Exploration-exploitation trade-off parameter
        self.actions: list[str] = actions # List of all possible actions

    def get_state_key(self, hunger: float, fatigue: float) -> str:
        """
        Converts continuous hunger and fatigue values into a discrete state key.

        This discretization simplifies the state space for the Q-table.
        For example, hunger 0.75 and fatigue 0.25 might become "h7_f2".

        Args:
            hunger (float): The agent's current hunger level (typically 0.0 to 1.0).
            fatigue (float): The agent's current fatigue level (typically 0.0 to 1.0).

        Returns:
            str: A string representation of the discretized state.
        """
        # Discretize hunger and fatigue into integer bins (0-9 range for simplicity).
        # Multiplying by 10 and taking int truncates the decimal.
        h_discrete = int(hunger * 10)
        f_discrete = int(fatigue * 10)
        return f"h{h_discrete}_f{f_discrete}"

    def choose_action(self, hunger: float, fatigue: float) -> str:
        """
        Selects an action based on the agent's current state using an epsilon-greedy policy.

        With probability `epsilon`, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value for the current state is chosen (exploitation).

        Args:
            hunger (float): The agent's current hunger level.
            fatigue (float): The agent's current fatigue level.

        Returns:
            str: The chosen action.
        """
        state_key = self.get_state_key(hunger, fatigue)
        if random.random() < self.epsilon:
            # Exploration: Choose a random action.
            return random.choice(self.actions)
        else:
            # Exploitation: Choose the action with the maximum Q-value for the current state.
            # `self.q_table[state_key]` returns the dictionary of Q-values for that state.
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, prev_hunger: float, prev_fatigue: float, action: str, reward: float, next_hunger: float, next_fatigue: float):
        """
        Updates the Q-value for a given state-action pair using the Q-learning formula.

        The Q-value represents the expected cumulative reward for taking a specific action
        in a given state and following the optimal policy thereafter.

        Args:
            prev_hunger (float): Hunger level before the action was taken.
            prev_fatigue (float): Fatigue level before the action was taken.
            action (str): The action that was performed.
            reward (float): The immediate reward received after performing the action.
            next_hunger (float): Hunger level after the action was taken.
            next_fatigue (float): Fatigue level after the action was taken.
        """
        prev_state_key = self.get_state_key(prev_hunger, prev_fatigue)
        next_state_key = self.get_state_key(next_hunger, next_fatigue)

        # Get the maximum Q-value for the next state (representing the best future action).
        max_future_q = max(self.q_table[next_state_key].values())
        # Get the current Q-value for the previous state and action.
        old_q = self.q_table[prev_state_key][action]

        # Q-learning formula:
        # New Q-value = Old Q-value + learning_rate * (Reward + discount_factor * Max_Future_Q - Old Q-value)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[prev_state_key][action] = new_q

    def __str__(self) -> str:
        """
        Provides a string representation of a sample of the Q-table.

        This is useful for debugging and inspecting the learning progress.

        Returns:
            str: A formatted string showing the first few entries of the Q-table.
        """
        # Display only the first 5 entries of the Q-table for brevity.
        return f"Q-table (sample): {dict(list(self.q_table.items())[:5])}"