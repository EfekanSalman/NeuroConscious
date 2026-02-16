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
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from agent.base_agent import Agent

class CognitiveModule(ABC):
    """
    Abstract base class for all plug-and-play cognitive modules.

    Each cognitive module represents a distinct mental faculty or processing unit
    that can influence the agent's internal state, perception, or decision-making.
    """
    def __init__(self, agent: 'Agent', name: str):
        """
        Initializes a cognitive module.

        Args:
            agent (Agent): The agent instance this module belongs to.
            name (str): The name of the cognitive module.
        """
        self.agent = agent
        self.name = name

    @abstractmethod
    def process(self) -> Dict[str, Any]:
        """
        Processes information and generates an output or influence.

        This method should be overridden by concrete cognitive modules to define
        their specific functionality (e.g., problem-solving, social reasoning).

        Returns:
            Dict[str, Any]: A dictionary containing any outputs or suggestions
                            from this module (e.g., {"suggested_action": "explore"}).
                            Returns an empty dict if no specific output.
        """
        pass

    def get_module_name(self) -> str:
        """Returns the name of the cognitive module."""
        return self.name

