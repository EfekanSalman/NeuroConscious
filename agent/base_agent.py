from core.state import InternalState
from core.motivation import MotivationEngine
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.reward_learner import RewardLearner
from core.learning.q_table_learner import QTableLearner

class Agent:
    def __init__(self, name: str, mood_strategy: MoodStrategy):
        self.name = name
        self.state = InternalState(mood_strategy)
        self.motivation = MotivationEngine(self.state)
        self.perception = {
            "food_available": False,
            "time_of_day": "day"
        }
        self.memory = {
            "food_last_seen": None
        }
        self.environment = None #Reference to the World
        self.current_step = 0
        self.episodic_memory = EpisodicMemory(capacity = 5)
        self.learner = RewardLearner()
        self._prev_state_snapshot = None
        self.q_learner = QTableLearner(actions=["seek_food", "rest", "explore"])
        self._prev_state_vals = None # Hunger/fatigue before action
        self._last_action = None

    def set_environment(self, env):
        self.environment = env

    def sense(self):
        """ Sense food availability and time of day. """
        if self.environment:
            self.perception["food_available"] = self.environment.food_available
            self.perception["time_of_day"] = self.environment.time_of_day
            self.current_step = self.environment.time_step

            # Remember the last time food was seen
            if self.environment.food_available:
                self.memory["food_last_seen"] = self.current_step

    def think(self):
        """
        Decide on an action using Q-learning.
        Also stores the current state for learning after action
        """
        # Update internal state based on time of day"
        delta = 1.5 if self.perception["time_of_day"] == "night" else 1.0
        self.state.update(delta_time=delta)

        # Save current hunger and fatigue before action (for Q-learning)
        self._prev_state_vals = (self.state.hunger, self.state.fatigue)

        # Choose an action using the Q-table with epsilon-greedy policy
        action = self.q_learner.choose_action(
            hunger = self.state.hunger,
            fatigue = self.state.fatigue
        )

        # Store chosen action for reward update
        self._last_action = action
        return action

    def act(self, action: str):
        """
        Perform the selected action and update Q-table with the reward.
        """
        # Take action and apply effects to internal state
        if action == "seek_food" and self.perception["food_available"]:
            print(f"{self.name} eats food.")
            self.state.hunger = max(0.0, self.state.hunger - 0.4)
        elif action == "seek_food":
            print(f"{self.name} searches for food but finds none.")
        elif action == "rest":
            print(f"{self.name} takes a rest.")
            self.state.fatigue = max(0.0, self.state.fatigue - 0.3)
        elif action == "explore":
            print(f"{self.name} is exploring...")

        # Learn from the action by updating Q-table
        if self._prev_state_vals and self._last_action:
            prev_h, prev_f = self._prev_state_vals
            next_h, next_f = self.state.hunger, self.state.fatigue

            # Define simple reward signal
            reward = 0.0
            if self._last_action == "seek_food":
                reward = max(0, prev_h - next_h)
            elif self._last_action == "rest":
                reward = max(0, prev_f - next_f)
            elif self._last_action == "explore":
                reward = -0.01 # Small penalty for wasting time

            self.q_learner.update(
                prev_hunger = prev_h,
                prev_fatigue = prev_f,
                action = self._last_action,
                reward = reward,
                next_hunger = next_h,
                next_fatigue = next_f
            )

        # Store in episodic memory
        self.episodic_memory.add(
            step = self.current_step,
            perception = self.perception,
            state = self.state,
            action=action
        )

    def log_status(self):
        print(f"{self.name} → {self.state}")
        print("Recent episodes:")
        print(self.episodic_memory)
        print("Q-table (sample):")
        print(self.q_learner)
