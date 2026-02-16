"""Internal physiological state of an agent."""

from config.constants import DEFAULT_HUNGER_INCREASE, DEFAULT_FATIGUE_INCREASE
from core.mood.base import MoodStrategy


class InternalState:
    """Tracks hunger, fatigue, and mood for an agent.

    Hunger and fatigue rise over time; mood is recomputed each tick
    via the injected MoodStrategy.
    """

    def __init__(self, mood_strategy: MoodStrategy, hunger: float = 0.5, fatigue: float = 0.5):
        self.hunger = hunger
        self.fatigue = fatigue
        self.mood: str = "neutral"
        self.mood_strategy = mood_strategy

    def update(self, delta_time: float = 1.0) -> None:
        """Advance physiological state by one tick.

        Args:
            delta_time: Time multiplier (e.g. 1.5 at night).
        """
        self.hunger = min(1.0, self.hunger + DEFAULT_HUNGER_INCREASE * delta_time)
        self.fatigue = min(1.0, self.fatigue + DEFAULT_FATIGUE_INCREASE * delta_time)
        self.mood = self.mood_strategy.calculate_mood(self.hunger, self.fatigue)

    def snapshot(self) -> "InternalState":
        """Create an independent copy of the current state."""
        clone = InternalState(self.mood_strategy)
        clone.hunger = self.hunger
        clone.fatigue = self.fatigue
        clone.mood = self.mood
        return clone

    def __str__(self) -> str:
        return f"Hunger: {self.hunger:.2f}, Fatigue: {self.fatigue:.2f}, Mood: {self.mood}"