from abc import ABC, abstractmethod

class MoodStrategy(ABC):
    """
    Abstract base class for defining different mood calculation strategies.

    This ABC provides a blueprint for how an agent's overall mood or emotional
    state should be derived from its internal physiological states, such as
    hunger and fatigue. Concrete implementations of this strategy will
    determine the specific rules for mood calculation.
    """
    @abstractmethod
    def calculate_mood(self, hunger: float, fatigue: float) -> str:
        """
        Abstract method to calculate the agent's current mood.

        Concrete strategy implementations must override this method to define
        their specific logic for determining mood based on internal states.

        Args:
            hunger (float): The agent's current hunger level (typically 0.0 to 1.0).
            fatigue (float): The agent's current fatigue level (typically 0.0 to 1.0).

        Returns:
            str: A string representing the calculated mood (e.g., "happy", "neutral", "sad").
        """
        pass