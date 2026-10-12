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

from core.mood.base import MoodStrategy
from typing import Dict, Any

class InternalState:
    """
    Manages the internal physiological and emotional state of an agent.

    This includes core needs like hunger and fatigue, and a calculated mood value.
    This version adds 'thirst' as a new physiological need.
    """
    def __init__(self, mood_strategy: MoodStrategy):
        """
        Initializes the agent's internal state.

        Args:
            mood_strategy (MoodStrategy): The strategy used to calculate the agent's mood.
        """
        self.hunger: float = 0.0    # 0.0 (full) to 1.0 (starving)
        self.fatigue: float = 0.0   # 0.0 (rested) to 1.0 (exhausted)
        self.thirst: float = 0.0    # New: 0.0 (hydrated) to 1.0 (dehydrated)
        self.mood_strategy: MoodStrategy = mood_strategy
        self.mood_value: float = 0.0 # -1.0 (very bad) to 1.0 (very good)

        # Initial mood calculation
        self._update_mood()

    def update(self, delta_time: float = 1.0):
        """
        Updates the agent's internal state over time.

        Args:
            delta_time (float): The amount of time that has passed, influencing
                                the rate of hunger and fatigue increase.
        """
        # Hunger and fatigue increase over time
        self.hunger = min(1.0, self.hunger + (0.01 * delta_time))
        self.fatigue = min(1.0, self.fatigue + (0.015 * delta_time))
        self.thirst = min(1.0, self.thirst + (0.02 * delta_time)) # New: Thirst increases over time

        # Recalculate mood based on updated internal states
        self._update_mood()

    def _update_mood(self):
        """
        Calculates and updates the agent's mood based on its current hunger,
        fatigue, and thirst levels using the assigned mood strategy.
        """
        self.mood_value = self.mood_strategy.calculate_mood(self.hunger, self.fatigue, self.thirst) # New: Pass thirst to mood strategy

    def snapshot(self) -> Dict[str, float]:
        """
        Returns a snapshot of the current internal state values.

        Returns:
            Dict[str, float]: A dictionary containing hunger, fatigue, and mood_value.
        """
        return {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "thirst": self.thirst, # New: Include thirst in snapshot
            "mood_value": self.mood_value
        }

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the internal state.
        """
        return (f"Hunger: {self.hunger:.2f}, Fatigue: {self.fatigue:.2f}, "
                f"Thirst: {self.thirst:.2f}, Mood: {self.mood_value:.2f}") # New: Include thirst in string representation


