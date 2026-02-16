"""Emotion subsystem."""

from core.emotion.emotion_state import EmotionState
from core.emotion.base_emotion import EmotionStrategy
from core.emotion.basic_emotion import BasicEmotionStrategy

__all__ = ["EmotionState", "EmotionStrategy", "BasicEmotionStrategy"]
