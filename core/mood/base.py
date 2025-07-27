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

class MoodStrategy(ABC):
    """
    Abstract base class for defining different mood calculation strategies.

    Concrete implementations of this class will provide specific logic
    for how an agent's mood is determined based on its internal state.
    This version includes 'thirst' as a parameter for mood calculation.
    """
    @abstractmethod
    def calculate_mood(self, hunger: float, fatigue: float, thirst: float) -> float:
        """
        Calculates the agent's mood based on its internal physiological states.

        Args:
            hunger (float): The agent's current hunger level (0.0 to 1.0).
            fatigue (float): The agent's current fatigue level (0.0 to 1.0).
            thirst (float): The agent's current thirst level (0.0 to 1.0).

        Returns:
            float: A numerical representation of the agent's mood (e.g., -1.0 for very bad, 1.0 for very good).
        """
        pass