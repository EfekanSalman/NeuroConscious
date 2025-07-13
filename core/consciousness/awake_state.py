from core.consciousness.base import ConsciousnessState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.base_agent import Agent

class AwakeState(ConsciousnessState):
    """
    Represents the default awake and active state of the agent.

    In this state, the agent performs its regular sensing, thinking, and acting
    based on its internal needs, perceptions, and learned behaviors.
    """
    def __init__(self, agent: 'Agent'):
        super().__init__(agent)

    def get_state_name(self) -> str:
        return "Awake"

    def enter(self):
        self.agent.internal_monologue += "Entering Awake state. Ready to engage with the world. "
        # Reset any specific state variables if needed
        self.agent.attention_focus = None # Diffuse attention in awake state

    def exit(self):
        self.agent.internal_monologue += "Exiting Awake state. "

    def sense(self):
        self.agent._sense_default()

    def think(self) -> str:
        # Awake state typically uses a deliberative mode unless critical needs arise
        decision_mode = self.get_decision_mode() # Get the decision mode for this state
        return self.agent._think_default(decision_mode) # Pass the decision_mode to _think_default

    def act(self, action: str):
        self.agent._act_default(action)

    def get_decision_mode(self) -> str:
        """
        Awake state typically uses a deliberative mode, but can become reactive
        if critical needs are very high.
        """
        if self.agent.internal_state.hunger > 0.8 or self.agent.internal_state.fatigue > 0.8:
            return 'reactive'
        return 'deliberative'

