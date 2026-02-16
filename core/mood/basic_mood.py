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
from config.constants import MOOD_HAPPY_THRESHOLD, MOOD_SAD_THRESHOLD

from core.mood.base import MoodStrategy

class BasicMoodStrategy(MoodStrategy):
    """
    A basic implementation of the MoodStrategy.

    This strategy calculates mood based on simple linear relationships
    with hunger, fatigue, and now, thirst.
    """
    def calculate_mood(self, hunger: float, fatigue: float, thirst: float) -> float:
        """
        Calculates the agent's mood based on hunger, fatigue, and thirst.

        Mood is a composite score where:
        - Low hunger contributes positively.
        - Low fatigue contributes positively.
        - Low thirst contributes positively.

        Args:
            hunger (float): The agent's current hunger level (0.0 to 1.0).
            fatigue (float): The agent's current fatigue level (0.0 to 1.0).
            thirst (float): The agent's current thirst level (0.0 to 1.0).

        Returns:
            float: A numerical representation of the agent's mood.
                   -1.0 represents a very bad mood, 1.0 represents a very good mood.
        """
        # Invert hunger, fatigue, and thirst values so that lower values (less need)
        # contribute positively to mood, and higher values (more need) contribute negatively.
        # Example: hunger 0.0 -> 1.0 (positive contribution), hunger 1.0 -> 0.0 (negative contribution)
        inverted_hunger = 1.0 - hunger
        inverted_fatigue = 1.0 - fatigue
        inverted_thirst = 1.0 - thirst

        # Assign weights to each factor. These weights can be tuned.
        # Hunger and thirst might have a stronger negative impact than fatigue.
        hunger_weight = 0.4
        fatigue_weight = 0.3
        thirst_weight = 0.3

        # Calculate a raw mood score.
        # Multiply inverted values by their weights.
        raw_mood = (inverted_hunger * hunger_weight +
                    inverted_fatigue * fatigue_weight +
                    inverted_thirst * thirst_weight)

        # Normalize the raw mood to a range of -1.0 to 1.0.
        # The sum of weights is 0.4 + 0.3 + 0.3 = 1.0.
        # So, raw_mood will be between 0.0 (all needs max) and 1.0 (all needs min).
        # To scale to -1.0 to 1.0: (raw_mood * 2) - 1.0
        normalized_mood = (raw_mood * 2.0) - 1.0

        return normalized_mood
