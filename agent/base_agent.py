from core.state import InternalState
from core.motivation import MotivationEngine
from core.mood.base import MoodStrategy

class Agent:
    def __init__(self, name: str, mood_strategy: MoodStrategy):
        self.name = name
        self.state = InternalState(mood_strategy)
        self.motivation = MotivationEngine(self.state)

    def sense(self):
        """ Detect the environment """
        pass

    def think(self):
        """ Update the situation and decide """
        self.state.update()
        return self.motivation.decide_action()

    def act(self, action: str):
        """ Take action """
        if action == "seek_food":
            self.state.hunger = max(0.0, self.state.hunger - 0.4)
        elif action == "rest":
            self.state.fatigue = max(0.0, self.state.fatigue - 0.3)
            # TODO add explore

    def log_status(self):
        print(f"{self.name} -> {self.state}")
