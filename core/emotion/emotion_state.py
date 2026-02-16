"""Emotion state container — stores and clamps emotion values."""


class EmotionState:
    """Holds named emotion values clamped to [0.0, 1.0].

    Default emotions: joy, fear, curiosity, frustration.
    """

    def __init__(self):
        self.emotions: dict[str, float] = {
            "joy": 0.5,
            "fear": 0.0,
            "curiosity": 0.5,
            "frustration": 0.0,
        }

    def get(self, emotion_name: str) -> float:
        """Return value for the named emotion, defaulting to 0.0."""
        return self.emotions.get(emotion_name, 0.0)

    def set(self, emotion_name: str, value: float) -> None:
        """Set emotion value, clamped to [0.0, 1.0]."""
        self.emotions[emotion_name] = max(0.0, min(1.0, value))

    def as_dict(self) -> dict[str, float]:
        """Return a copy of all emotion values."""
        return dict(self.emotions)

    def __str__(self) -> str:
        return ", ".join(f"{k}: {v:.2f}" for k, v in self.emotions.items())
