class EmotionState:
    def __init__(self):
        self.emotions = {
            "joy": 0.5,
            "fear": 0.0,
            "curiosity": 0.5,
            "frustration": 0.0
        }

    def get(self, emotion_name: str) -> float:
        return self.emotions.get(emotion_name, 0.0)

    def set(self, emotion_name: str, value: float):
        self.emotions[emotion_name] = max(0.0, min(1.0, value))  # clamp between 0-1

    def all(self):
        return self.emotions

    def __str__(self):
        return ", ".join([f"{k}: {v:.2f}" for k, v in self.emotions.items()])
