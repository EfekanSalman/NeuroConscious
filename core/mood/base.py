from abc import ABC, abstractmethod

class MoodStrategy(ABC):
    @abstractmethod
    def calculate_mood(self, hunger: float, fatigue: float) -> str:
        """Calculate the emotional state."""
        pass