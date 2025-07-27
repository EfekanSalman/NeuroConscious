# !/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class EmotionState:
    """
    Represents the emotional state of an agent.

    This class holds the current values for various emotions (e.g., joy, fear, frustration, curiosity)
    and provides methods to update and retrieve them. Emotion values are typically normalized between 0.0 and 1.0.
    """
    def __init__(self, joy: float = 0.0, fear: float = 0.0, frustration: float = 0.0, curiosity: float = 0.0):
        """
        Initializes the agent's emotional state.

        Args:
            joy (float, optional): Initial joy level. Defaults to 0.0.
            fear (float, optional): Initial fear level. Defaults to 0.0.
            frustration (float, optional): Initial frustration level. Defaults to 0.0.
            curiosity (float, optional): Initial curiosity level. Defaults to 0.0.
        """
        self._emotions = {
            "joy": self._clamp_value(joy),
            "fear": self._clamp_value(fear),
            "frustration": self._clamp_value(frustration),
            "curiosity": self._clamp_value(curiosity)
        }

    def _clamp_value(self, value: float) -> float:
        """Clamps an emotion value between 0.0 and 1.0."""
        return max(0.0, min(1.0, value))

    def get(self, emotion_name: str) -> float:
        """
        Retrieves the current value of a specific emotion.

        Args:
            emotion_name (str): The name of the emotion (e.g., "joy", "fear").

        Returns:
            float: The current value of the specified emotion. Returns 0.0 if emotion not found.
        """
        return self._emotions.get(emotion_name, 0.0)

    def set(self, emotion_name: str, value: float):
        """
        Sets the value of a specific emotion, clamping it between 0.0 and 1.0.

        Args:
            emotion_name (str): The name of the emotion.
            value (float): The new value for the emotion.
        """
        if emotion_name in self._emotions:
            self._emotions[emotion_name] = self._clamp_value(value)
        else:
            print(f"Warning: Attempted to set unknown emotion '{emotion_name}'.")

    def update_emotion(self, emotion_name: str, delta: float):
        """
        Updates an emotion by adding a delta value, clamping the result.

        Args:
            emotion_name (str): The name of the emotion to update.
            delta (float): The value to add to the current emotion level.
        """
        current_value = self.get(emotion_name)
        self.set(emotion_name, current_value + delta)

    def get_emotions(self) -> dict[str, float]:
        """
        Returns a dictionary of all current emotion values.

        Returns:
            dict[str, float]: A dictionary where keys are emotion names and values are their current levels.
        """
        return self._emotions.copy() # Return a copy to prevent external modification

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the emotional state.
        """
        return ", ".join([f"{name.capitalize()}: {value:.2f}" for name, value in self._emotions.items()])

