#!/usr/bin/env python3
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

from __future__ import annotations
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from agent.base_agent import Agent
    from core.communication.communication_manager import Message


class MoodStrategy:
    """
    Manages the agent's emotional state or mood.
    The mood can change based on the content of incoming messages.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the mood strategy with a default mood.
        """
        self.agent = agent
        self.mood_state = "neutral"  # The agent's current mood.
        self.mood_history = []  # A log of past mood states.

    def update_mood(self, message: Message):
        """
        Updates the agent's mood based on the content of the incoming message.
        This is a simple implementation using keyword detection.

        Args:
            message (Message): The message object that is influencing the mood.
        """
        print(f"[{self.agent.name}]: Analyzing message to update mood...")

        content = message.content.lower()
        new_mood = self.mood_state  # Default to current mood

        # Simple keyword-based mood detection
        positive_keywords = ["great", "good", "happy", "thanks", "harika", "güzel"]
        negative_keywords = ["bad", "terrible", "annoying", "problem", "kötü", "sorun"]

        if any(word in content for word in positive_keywords):
            new_mood = "happy"
        elif any(word in content for word in negative_keywords):
            new_mood = "annoyed"
        else:
            # If no strong keywords, there's a small chance of a mood shift
            if random.random() < 0.2:
                new_mood = random.choice(["calm", "curious"])

        if new_mood != self.mood_state:
            print(f"[{self.agent.name}]: Mood changed from '{self.mood_state}' to '{new_mood}'")
            self.mood_history.append(self.mood_state)
            self.mood_state = new_mood
            self.agent.internal_monologue += f" My mood has shifted to {self.mood_state}."

        print(f"[{self.agent.name}]: Current mood is '{self.mood_state}'")

    def get_mood_state(self) -> str:
        """
        Returns the agent's current mood state.
        """
        return self.mood_state

