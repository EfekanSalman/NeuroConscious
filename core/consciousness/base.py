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
# We will need to import Agent later, but to avoid circular imports,
# we'll use type hints and import it inside methods if absolutely necessary,
# or pass necessary data directly. For now, just ABC.

class ConsciousnessState(ABC):
    """
    Abstract base class for different states of consciousness.

    This class defines the interface for how an agent behaves (senses, thinks, acts)
    under different states of consciousness. Each concrete state will implement
    these methods differently.
    """
    def __init__(self, agent):
        """
        Initializes the consciousness state with a reference to the agent.

        Args:
            agent: The agent instance this state belongs to.
        """
        self.agent = agent

    @abstractmethod
    def enter(self):
        """
        Method called when the agent enters this state of consciousness.
        Can be used for initialization or state-specific setup.
        """
        pass

    @abstractmethod
    def exit(self):
        """
        Method called when the agent exits this state of consciousness.
        Can be used for cleanup or state-specific teardown.
        """
        pass

    @abstractmethod
    def sense(self):
        """
        Defines how the agent perceives the environment in this state.
        """
        pass

    @abstractmethod
    def think(self) -> str:
        """
        Defines how the agent processes information and decides an action in this state.

        Returns:
            str: The chosen action.
        """
        pass

    @abstractmethod
    def act(self, action: str):
        """
        Defines how the agent performs an action in this state.

        Args:
            action (str): The action to perform.
        """
        pass

    @abstractmethod
    def get_state_name(self) -> str:
        """
        Returns the name of this consciousness state.
        """
        pass

