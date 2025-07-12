# core/consciousness/base.py

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

