"""Abstract base for emotion strategies."""

from abc import ABC, abstractmethod
from core.emotion.emotion_state import EmotionState


class EmotionStrategy(ABC):
    """Base class for all emotion computation strategies."""

    def __init__(self, emotion_state: EmotionState):
        self.emotion_state = emotion_state

    @abstractmethod
    def update_emotions(self, perception: dict, internal_state) -> None:
        """Recalculate emotion values from perception and internal state."""
        pass
