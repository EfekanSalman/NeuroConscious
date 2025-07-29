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
from typing import TYPE_CHECKING, List, Dict, Any
if TYPE_CHECKING:
    from agent.base_agent import Agent

class BasePerceptionManager(ABC):
    """
    Abstract base class for all perception managers.

    Defines the interface for how an agent perceives its environment.
    Concrete implementations will handle specific sensing mechanisms,
    sensory noise, and attention modulation.
    """
    def __init__(self, agent: 'Agent'):
        """
        Initializes the base perception manager with a reference to the agent.

        Args:
            agent (Agent): The agent instance this manager belongs to.
        """
        self.agent = agent

    @abstractmethod
    def update_perception(self):
        """
        Updates the agent's perception dictionary based on the current environment state.
        This method should encapsulate all sensing logic.
        """
        pass

    @abstractmethod
    def _get_local_grid_view(self, radius: int) -> List[List[str]]:
        """
        Retrieves a local view of the grid around the agent.

        Args:
            radius (int): The radius of the square view.

        Returns:
            List[List[str]]: A 2D list representing the agent's local view.
        """
        pass

