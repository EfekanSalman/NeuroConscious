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

from core.consciousness.base import ConsciousnessState

class AsleepState(ConsciousnessState):
    """
    Represents the agent's sleeping state of consciousness.

    In this state, the agent's sensing and acting capabilities are severely limited,
    and its primary focus is on resting and recovering fatigue.
    """
    def enter(self):
        """
        Actions to perform when entering the Asleep state.
        """
        print(f"{self.agent.name} is now Asleep.")
        # Reset attention focus when going to sleep
        self.agent.attention_focus = None
        # Internal monologue for entering sleep
        self.agent.internal_monologue += "I am feeling very tired. I need to rest and recover. Entering sleep state. "


    def exit(self):
        """
        Actions to perform when exiting the Asleep state.
        """
        print(f"{self.agent.name} is no longer Asleep.")
        self.agent.internal_monologue += "I am waking up, feeling more rested. "

    def sense(self):
        """
        Agent's sensing is minimal in the Asleep state.
        It might only perceive critical environmental changes (e.g., extreme weather)
        or internal needs (e.g., critical hunger).
        """
        # Minimal sensing: only update time of day and critical hunger/fatigue
        if self.agent.environment:
            self.agent.perception["time_of_day"] = self.agent.environment.time_of_day
            self.agent.perception["current_weather"] = self.agent.environment.current_weather
            # Simulate very low perception accuracy
            self.agent.perception["food_in_sight"] = False
            self.agent.perception["other_agents_in_sight"] = []
            self.agent.internal_monologue += "While asleep, my perceptions are minimal. "

    def think(self) -> str:
        """
        Agent's thinking in the Asleep state is primarily focused on resting.
        It will always choose the 'rest' action if possible.
        """
        # Always choose to rest if asleep
        self.agent.internal_monologue += "While asleep, my only thought is to continue resting. "
        return "rest"

    def act(self, action: str):
        """
        Agent's actions in the Asleep state are limited to resting.
        Other actions are ignored or have no effect.

        Args:
            action (str): The action to perform.
        """
        if action == "rest":
            # Directly apply rest effects, bypassing normal act logic for simplicity in sleep state
            self.agent.internal_state.fatigue = max(0.0, self.agent.internal_state.fatigue - 0.7) # More effective rest
            self.agent.last_action_reward = 0.6 # Higher reward for effective rest
            print(f"{self.agent.name} is deeply resting.")
            self.agent.internal_monologue += "I am resting deeply. "
        else:
            print(f"{self.agent.name} tried to {action} while asleep, but only rested.")
            self.agent.internal_monologue += f"I tried to {action} in my sleep, but I'm still resting. "
            # Still apply some rest effect even if trying other action
            self.agent.internal_state.fatigue = max(0.0, self.agent.internal_state.fatigue - 0.2)
            self.agent.last_action_reward = 0.1 # Small reward for partial rest

        # Update physiological state (hunger might still increase, but fatigue decreases)
        # This part is crucial for waking up conditions.
        self.agent.internal_state.update(delta_time=1.0)
        # Emotions might still update based on internal state changes
        self.agent.emotion_strategy.update_emotions(self.agent.perception, self.agent.internal_state)

        # No DQN update or episodic memory add in sleep state for simplicity,
        # as learning is not the primary goal here.
        # If you want to learn from sleep, you'd integrate it.

    def get_state_name(self) -> str:
        """
        Returns the name of this consciousness state.
        """
        return "Asleep"

