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

from collections import deque
from core.emotion.emotion_state import EmotionState

class EpisodicMemory:
    """
    Manages a limited capacity episodic memory for an agent.

    This class uses a deque (double-ended queue) to store a fixed number of
    recent experiences (episodes), providing the agent with a short-term
    recollection of past perceptions, internal states, and actions.
    It now also incorporates emotional weighting to remember experiences more vividly.
    """
    def __init__(self, capacity: int = 5):
        """
        Initializes the episodic memory with a specified capacity.

        Args:
            capacity (int): The maximum number of episodes to store in memory.
                            When new episodes are added and the capacity is reached,
                            the oldest episode is automatically discarded. Defaults to 5.
        """
        self.capacity: int = capacity
        # deque is used for efficient appending and popping from both ends,
        # and 'maxlen' automatically handles capacity limits.
        self.memory: deque = deque(maxlen=capacity)

    def _calculate_emotional_weight(self, emotions: EmotionState) -> float:
        """
        Calculates an emotional weight for an episode based on current emotion levels.

        This simple heuristic assigns a higher weight to experiences associated
        with strong positive (joy) or negative (fear, frustration) emotions.
        The weight can be used to influence recall priority or memory consolidation.
        A higher value indicates a more "memorable" or "impactful" experience.

        Args:
            emotions (EmotionState): The agent's current emotional state.

        Returns:
            float: A calculated emotional weight (typically between 0.0 and 1.0, but can exceed 1.0).
        """
        # Define how much each emotion contributes to the memory's weight.
        # These coefficients can be tuned for different effects.
        joy_contribution = emotions.get("joy") * 0.3
        fear_contribution = emotions.get("fear") * 0.6  # Fear often leads to stronger, more salient memories.
        frustration_contribution = emotions.get("frustration") * 0.4
        curiosity_contribution = emotions.get("curiosity") * 0.1

        # Sum of weighted emotional contributions.
        total_weight = joy_contribution + fear_contribution + frustration_contribution + curiosity_contribution

        # Optionally, clamp the weight to a maximum value (e.g., 1.0) or allow it to exceed 1.0
        # for a broader range of memorability. Here, we clamp it for simplicity.
        return min(1.0, total_weight)

    def add(self, step: int, perception: dict, internal_state, action: str, emotions: EmotionState):
        """
        Adds a new episode (experience) to the agent's episodic memory,
        including an emotional weight.

        Each episode captures a snapshot of the agent's situation at a specific
        time step, including its perceptions, internal physiological state,
        the action it performed, and a calculated emotional weight for that moment.

        Args:
            step (int): The current simulation time step.
            perception (dict): A dictionary representing the agent's sensory
                               input and environmental observations at this step.
            internal_state: The agent's internal state object (e.g., InternalState instance),
                   containing attributes like 'hunger', 'fatigue', and 'mood'.
                   We use 'internal_state' here for clarity as per our earlier refactor.
            action (str): The action performed by the agent at this step.
            emotions (EmotionState): The agent's emotional state at the time the episode occurred.
                                     Used to calculate the emotional weight of this memory.
        """
        # Calculate the emotional significance of this specific moment.
        emotional_weight = self._calculate_emotional_weight(emotions)

        episode = {
            "step": step,
            "perception": perception.copy(), # Store a copy to prevent external modifications
            "state": { # Snapshot of the internal physiological state at that time
                "hunger": round(internal_state.hunger, 2),
                "fatigue": round(internal_state.fatigue, 2),
                # FIX: Changed to use internal_state.mood_value (float) instead of internal_state.mood (string)
                # This resolves the AttributeError if 'mood' property is somehow not accessible.
                "mood_value": round(internal_state.mood_value, 2)
            },
            "action": action,
            "emotional_weight": round(emotional_weight, 2) # New field: Emotional weight for this episode
        }
        self.memory.append(episode)

    def get_memory(self) -> list[dict]:
        """
        Retrieves all currently stored episodes in the episodic memory.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an episode.
                        The episodes are ordered from oldest to newest.
        """
        return list(self.memory)

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the episodic memory content.

        This method is useful for quickly reviewing the agent's recent history
        for logging or debugging purposes, now including the emotional weight.

        Returns:
            str: A multi-line string, with each line detailing a remembered episode's
                 step, action, mood, and emotional weight.
        """
        # FIX: Changed to display 'mood_value' and formatted it as a float.
        return "\n".join([f"Step {ep['step']}: {ep['action']}, Mood Value: {ep['state']['mood_value']:.2f}, Weight: {ep['emotional_weight']:.2f}" for ep in self.memory])

