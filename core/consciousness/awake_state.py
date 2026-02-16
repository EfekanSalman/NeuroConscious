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

class AwakeState(ConsciousnessState):
    """
    Represents the agent's normal, fully awake state of consciousness.

    In this state, the agent performs its regular sensing, thinking, and acting
    with full capabilities, as defined by its core Agent class methods.
    """
    def enter(self):
        """
        Actions to perform when entering the Awake state.
        """
        # print(f"{self.agent.name} is now Awake.")
        pass

    def exit(self):
        """
        Actions to perform when exiting the Awake state.
        """
        # print(f"{self.agent.name} is no longer Awake.")
        pass

    def sense(self):
        """
        Agent senses normally in the Awake state.
        Delegates to the agent's default sense method.
        """
        self.agent._sense_default() # Call the agent's internal default sense method

    def think(self) -> str:
        """
        Agent thinks normally in the Awake state.
        Delegates to the agent's default think method.
        In the Awake state, the agent uses 'deliberative' decision mode by default.

        Returns:
            str: The chosen action.
        """
        # FIX: Pass 'deliberative' as the decision_mode to _think_default
        return self.agent._think_default(decision_mode="deliberative")

    def act(self, action: str):
        """
        Agent acts normally in the Awake state.
        Delegates to the agent's default act method.

        Args:
            action (str): The action to perform.
        """
        self.agent._act_default(action) # Call the agent's internal default act method

    def get_state_name(self) -> str:
        """
        Returns the name of this consciousness state.
        """
        return "Awake"
