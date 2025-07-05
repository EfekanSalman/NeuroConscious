from core.mood.base import MoodStrategy
from config.constants import MOOD_HAPPY_THRESHOLD, MOOD_SAD_THRESHOLD

class BasicMoodStrategy(MoodStrategy):
    """
    Implements a basic strategy for calculating an agent's overall mood.

    This strategy categorizes the agent's mood as 'happy', 'sad', or 'neutral'
    based on predefined thresholds for hunger and fatigue levels. It provides
    a straightforward mapping from physiological states to emotional states.
    """
    def calculate_mood(self, hunger: float, fatigue: float) -> str:
        """
        Calculates the agent's current mood based on its hunger and fatigue levels.

        The mood is determined by comparing the agent's current hunger and fatigue
        against predefined thresholds (MOOD_HAPPY_THRESHOLD, MOOD_SAD_THRESHOLD).

        Args:
            hunger (float): The agent's current hunger level (typically 0.0 to 1.0).
            fatigue (float): The agent's current fatigue level (typically 0.0 to 1.0).

        Returns:
            str: The calculated mood: "sad", "happy", or "neutral".
                 - "sad": If either hunger or fatigue is above the MOOD_SAD_THRESHOLD.
                 - "happy": If both hunger and fatigue are below the MOOD_HAPPY_THRESHOLD.
                 - "neutral": In all other cases.
        """
        # Determine if the agent is 'sad'.
        # A high level of hunger OR fatigue indicates a sad mood.
        if hunger > MOOD_SAD_THRESHOLD or fatigue > MOOD_SAD_THRESHOLD:
            return "sad"
        # Determine if the agent is 'happy'.
        # Both hunger AND fatigue must be low to indicate a happy mood.
        elif hunger < MOOD_HAPPY_THRESHOLD and fatigue < MOOD_HAPPY_THRESHOLD:
            return "happy"
        # If neither 'sad' nor 'happy' conditions are met, the agent is 'neutral'.
        else:
            return "neutral"