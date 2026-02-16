from config.constants import DEFAULT_HUNGER_INCREASE, DEFAULT_FATIGUE_INCREASE
from core.mood.base import MoodStrategy

class InternalState:
    def __init__(self, mood_strategy: MoodStrategy, hunger: float = 0.5, fatigue: float = 0.5):
        self.hunger = hunger
        self.fatigue = fatigue
        self.mood = "neutral"
        self.mood_strategy = mood_strategy

    def update(self, delta_time: float = 1.0):
        """Hunger and fatigue increase, mood recalculates."""
        self.hunger = min(1.0, self.hunger + DEFAULT_HUNGER_INCREASE * delta_time)
        self.fatigue = min(1.0, self.fatigue + DEFAULT_FATIGUE_INCREASE * delta_time)
        self.mood = self.mood_strategy.calculate_mood(self.hunger, self.fatigue)

    # inside InternalState
    def snapshot(self):
        return type(self)(
            self.mood_strategy
        )._copy_values_from(self)

    def _copy_values_from(self, other):
        self.hunger = other.hunger
        self.fatigue = other.fatigue
        self.mood = other.mood
        return self

    def __str__(self):
        return f"Hunger: {self.hunger:.2f}, Fatigue: {self.fatigue:.2f}, Mood: {self.mood}"