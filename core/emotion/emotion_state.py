class EmotionState:
    """
    Manages the current emotional state of an agent.

    This class stores various emotion levels (e.g., joy, fear, curiosity, frustration)
    and provides methods to access, update, and represent these emotions.
    Emotion values are clamped between 0.0 and 1.0 to ensure valid ranges.
    """
    def __init__(self):
        """
        Initializes the EmotionState with default emotion levels.

        The initial state provides a neutral starting point for the agent's emotional well-being.
        """
        self.emotions: dict[str, float] = {
            "joy": 0.5,          # Represents happiness or contentment.
            "fear": 0.0,         # Represents apprehension or dread.
            "curiosity": 0.5,    # Represents a desire to explore or learn.
            "frustration": 0.0   # Represents annoyance or dissatisfaction.
        }

    def get(self, emotion_name: str) -> float:
        """
        Retrieves the current value of a specific emotion.

        Args:
            emotion_name (str): The name of the emotion to retrieve (e.g., "joy", "fear").

        Returns:
            float: The current value of the emotion. Returns 0.0 if the emotion name is not found.
        """
        return self.emotions.get(emotion_name, 0.0)

    def set(self, emotion_name: str, value: float):
        """
        Sets the value of a specific emotion, clamping it between 0.0 and 1.0.

        This ensures that emotion levels remain within a realistic and manageable range.

        Args:
            emotion_name (str): The name of the emotion to set.
            value (float): The new value for the emotion. Will be clamped between 0.0 and 1.0.
        """
        # Clamps the value to be between 0.0 (minimum) and 1.0 (maximum).
        self.emotions[emotion_name] = max(0.0, min(1.0, value))

    def all(self) -> dict:
        """
        Returns a dictionary containing all current emotion names and their values.

        Returns:
            dict: A dictionary where keys are emotion names (str) and values are their levels (float).
        """
        return self.emotions

    def __str__(self) -> str:
        """
        Provides a string representation of the agent's current emotional state.

        This method is useful for logging and debugging, offering a quick overview of emotions.

        Returns:
            str: A comma-separated string listing each emotion and its value, formatted to two decimal places.
                 Example: "joy: 0.50, fear: 0.10"
        """
        return ", ".join([f"{k}: {v:.2f}" for k, v in self.emotions.items()])