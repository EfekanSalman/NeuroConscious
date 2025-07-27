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

from abc import ABC, abstractmethod
from core.emotion.emotion_state import EmotionState

class EmotionStrategy(ABC):
    """
    Abstract base class (ABC) for defining different emotion calculation strategies.

    This class serves as a blueprint for how an agent's emotions should be
    updated based on its perceptions and internal states. Concrete implementations
    must inherit from this class and provide their own logic for the
    'update_emotions' method.
    """
    def __init__(self, emotion_state: EmotionState):
        """
        Initializes the EmotionStrategy with a reference to the agent's emotional state.

        Args:
            emotion_state (EmotionState): An instance of EmotionState to be managed
                                          and updated by this strategy.
        """
        self.emotion_state: EmotionState = emotion_state

    @abstractmethod
    def update_emotions(self, perception: dict, internal_state):
        """
        Abstract method to update the agent's emotions based on current perception
        and internal physiological state.

        Concrete strategy implementations must override this method to define
        their specific emotion calculation logic.

        Args:
            perception (dict): A dictionary containing the agent's current sensory
                               inputs and environmental observations (e.g., food_available, time_of_day).
            internal_state: An object representing the agent's current internal
                            physiological states (e.g., hunger, fatigue, mood).
                            Expected to be an instance of InternalState or similar.
        """
        pass