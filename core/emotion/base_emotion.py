from abc import ABC, abstractmethod
from core.emotion.emotion_state import EmotionState

class EmotionStrategy(ABC):
    def __init__(self, emotion_state: EmotionState):
        self.emotion_state = emotion_state

    @abstractmethod
    def update_emotions(self, perception: dict, internal_state):
        pass
