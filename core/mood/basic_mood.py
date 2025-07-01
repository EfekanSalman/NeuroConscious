from core.mood.base import MoodStrategy
from config.constants import MOOD_HAPPY_THRESHOLD, MOOD_SAD_THRESHOLD

class BasicMoodStrategy(MoodStrategy):
    def calculate_mood(self, hunger: float, fatigue: float) -> str:
        if hunger > MOOD_SAD_THRESHOLD or fatigue > MOOD_SAD_THRESHOLD:
            return "sad"
        elif hunger < MOOD_HAPPY_THRESHOLD and fatigue < MOOD_HAPPY_THRESHOLD:
            return "happy"
        else:
            return "neutral"
