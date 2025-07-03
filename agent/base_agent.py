from core.state import InternalState
from core.motivation import MotivationEngine
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.reward_learner import RewardLearner

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
        """ Update internal state and decide what to do."""
        delta = 1.5 if self.perception["time_of_day"] == "night" else 1.0
        self.state.update(delta_time=delta)

        self._prev_state_snapshot = self.state.snapshot()  # store before acting
        learned_action = self.learner.get_best_action(
            self.state.hunger, self.state.fatigue
        )

        if learned_action:
            return learned_action

        return self.motivation.decide_action(
            perception=self.perception, memory=self.memory
        )

    def act(self, action: str):
        """Perform the selected action."""
        # Perform effects
        if action == "seek_food" and self.perception["food_available"]:
            print(f"{self.name} eats food.")
            self.state.hunger = max(0.0, self.state.hunger - 0.4)
        elif action == "seek_food":
            print(f"{self.name} searches for food but finds none.")
        elif action == "rest":
            print(f"{self.name} takes a rest.")
            self.state.fatigue = max(0.0, self.state.fatigue - 0.3)

        # Update reward learner
        if self._prev_state_snapshot:
            self.learner.update(self._prev_state_snapshot, self.state, action)

        self.episodic_memory.add(
            step=self.current_step,
            perception=self.perception,
            state=self.state,
            action=action
        )

    def log_status(self):
        print(f"{self.name} -> {self.state} | Memory: {self.memory}")
