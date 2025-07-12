# core/consciousness/awake_state.py

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

        Returns:
            str: The chosen action.
        """
        return self.agent._think_default() # Call the agent's internal default think method

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

