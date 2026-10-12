# !/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from collections import defaultdict
from typing import List, Dict
import json

class QTableLearner:
    """
    Implements a Q-learning agent that learns optimal actions based on states and rewards.

    Q-learning is a model-free reinforcement learning algorithm to learn a policy
    telling an agent what action to take under what circumstances. It uses a Q-table
    to store the 'quality' (expected utility) of taking a given action in a given state.
    This version includes epsilon decay for adaptive exploration and supports
    saving/loading the Q-table for persistent learning.
    """
    def __init__(self, actions: List[str], alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay_rate: float = 0.995):
        """
        Initializes the QTableLearner.

        Args:
            actions (List[str]): A list of possible actions the agent can take (e.g., ["seek_food", "rest"]).
            alpha (float, optional): The learning rate (alpha). Controls how much new information
                                     overrides old information. Defaults to 0.1.
            gamma (float, optional): The discount factor (gamma). Determines the importance of
                                     future rewards. Defaults to 0.9.
            epsilon (float, optional): The initial exploration rate (epsilon). The probability of
                                       the agent choosing a random action (exploration) instead of
                                       the best known action (exploitation). Defaults to 1.0 (start with full exploration).
            epsilon_min (float, optional): The minimum value epsilon can decay to. Defaults to 0.01.
            epsilon_decay_rate (float, optional): The rate at which epsilon decays after each update.
                                                  Defaults to 0.995 (epsilon * 0.995).
        """
        # Q-table stores Q-values for state-action pairs.
        # defaultdict is used so that new states are initialized with all actions having Q-value 0.0.
        self.q_table: defaultdict[str, Dict[str, float]] = defaultdict(lambda: {action: 0.0 for action in actions})
        self.alpha: float = alpha       # Learning rate
        self.gamma: float = gamma       # Discount factor for future rewards
        self.epsilon: float = epsilon   # Current exploration rate
        self.epsilon_min: float = epsilon_min # Minimum exploration rate
        self.epsilon_decay_rate: float = epsilon_decay_rate # Rate at which epsilon decreases
        self.actions: List[str] = actions # List of all possible actions

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
            # If all Q-values are 0.0 (e.g., new state), max() will pick one arbitrarily.
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, prev_hunger: float, prev_fatigue: float, action: str, reward: float, next_hunger: float, next_fatigue: float):
        """
        Updates the Q-value for a given state-action pair using the Q-learning formula.
        Also applies epsilon decay after the update.

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
        # If the next_state_key is new, self.q_table[next_state_key] will initialize with 0.0s.
        max_future_q = max(self.q_table[next_state_key].values())
        # Get the current Q-value for the previous state and action.
        old_q = self.q_table[prev_state_key][action]

        # Q-learning formula:
        # New Q-value = Old Q-value + learning_rate * (Reward + discount_factor * Max_Future_Q - Old Q-value)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[prev_state_key][action] = new_q

        # Apply epsilon decay
        # Decrease epsilon, but ensure it doesn't fall below epsilon_min.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def save_q_table(self, filepath: str):
        """
        Saves the current Q-table to a JSON file.

        Args:
            filepath (str): The path to the file where the Q-table will be saved.
        """
        # Convert defaultdict to a regular dict for JSON serialization
        # Nested dictionaries also need to be converted
        serializable_q_table = {
            state_key: dict(actions_dict)
            for state_key, actions_dict in self.q_table.items()
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_q_table, f, indent=2)
            print(f"Q-table saved to {filepath}")
        except IOError as e:
            print(f"Error saving Q-table to {filepath}: {e}")

    def load_q_table(self, filepath: str):
        """
        Loads a Q-table from a JSON file.

        Args:
            filepath (str): The path to the file from which the Q-table will be loaded.
        """
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            # Convert the loaded dict back to defaultdict
            self.q_table = defaultdict(lambda: {action: 0.0 for action in self.actions}, loaded_data)
            print(f"Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"No Q-table found at {filepath}. Starting with a new Q-table.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}. Starting with a new Q-table.")
        except IOError as e:
            print(f"Error loading Q-table from {filepath}: {e}. Starting with a new Q-table.")

    def __str__(self) -> str:
        """
        Provides a string representation of a sample of the Q-table.

        This is useful for debugging and inspecting the learning progress.

        Returns:
            str: A formatted string showing the first few entries of the Q-table.
        """
        # Display only the first 5 entries of the Q-table for brevity.
        # Also show the current epsilon value.
        return f"Q-table (sample): {dict(list(self.q_table.items())[:5])}, Epsilon: {self.epsilon:.3f}"

