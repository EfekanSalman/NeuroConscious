from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

# Circular import'ı önlemek için Agent'ı yalnızca tip denetimi için içe aktarın
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

