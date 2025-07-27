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

class RewardLearner:
    """
    A simple learner responsible for processing rewards based on state transitions.

    In a more complex system, this might involve learning a reward function
    or shaping rewards for other learning algorithms. This version can adjust
    rewards based on the agent's current mood.
    """
    def __init__(self):
        """
        Initializes the RewardLearner.
        """
        pass # Currently, no specific parameters are needed for initialization.

    def update(self, previous_state, current_state, action: str, raw_reward: float, mood_value: float):
        """
        Processes and potentially adjusts a raw reward based on the agent's mood.

        This method simulates how an agent's emotional state might influence its
        perception of rewards, making learning more emotionally nuanced.

        Args:
            previous_state: The agent's internal state before the action.
            current_state: The agent's internal state after the action.
            action (str): The action performed.
            raw_reward (float): The immediate reward received from the environment.
            mood_value (float): The current numerical mood value of the agent (0.0 to 1.0).
                                0.0 typically means very negative mood, 1.0 means very positive.
        """
        adjusted_reward = raw_reward

        # Simple mood-based reward adjustment:
        # Positive mood (mood_value > 0.5) might amplify positive rewards or dampen negative ones.
        # Negative mood (mood_value < 0.5) might dampen positive rewards or amplify negative ones.

        if mood_value > 0.7: # Very positive mood (e.g., happy)
            if raw_reward > 0:
                adjusted_reward *= 1.2 # Amplify positive rewards by 20%
                print(f"[REWARD LEARNER] Positive mood ({mood_value:.2f}) amplified reward: {raw_reward:.2f} -> {adjusted_reward:.2f}")
            elif raw_reward < 0:
                adjusted_reward *= 0.8 # Dampen negative rewards by 20%
                print(f"[REWARD LEARNER] Positive mood ({mood_value:.2f}) dampened penalty: {raw_reward:.2f} -> {adjusted_reward:.2f}")
        elif mood_value < 0.3: # Very negative mood (e.g., sad, frustrated)
            if raw_reward > 0:
                adjusted_reward *= 0.7 # Dampen positive rewards by 30%
                print(f"[REWARD LEARNER] Negative mood ({mood_value:.2f}) dampened reward: {raw_reward:.2f} -> {adjusted_reward:.2f}")
            elif raw_reward < 0:
                adjusted_reward *= 1.3 # Amplify negative rewards (penalties) by 30%
                print(f"[REWARD LEARNER] Negative mood ({mood_value:.2f}) amplified penalty: {raw_reward:.2f} -> {adjusted_reward:.2f}")
        else: # Neutral mood, no significant adjustment
            pass # Reward remains raw_reward

        return adjusted_reward


