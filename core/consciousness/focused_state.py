from core.consciousness.base import ConsciousnessState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.base_agent import Agent

class FocusedState(ConsciousnessState):
    """
    Represents a state where the agent is intensely focused on a specific task or goal.

    In this state, the agent's attention is narrowed, and its decision-making
    might be more goal-directed or reactive towards the focus.
    """
    def __init__(self, agent: 'Agent'):
        super().__init__(agent)

    def get_state_name(self) -> str:
        return "Focused"

    def enter(self):
        self.agent.internal_monologue += f"Entering Focused state. Concentrating on {self.agent.attention_focus}. "
        # Attention focus should be set before entering this state

    def exit(self):
        self.agent.internal_monologue += "Exiting Focused state. "
        # self.agent.attention_focus = None # Might clear attention focus on exit

    def sense(self):
        # Sensing might be biased towards the attention focus
        self.agent._sense_default() # For simplicity, use default sensing for now

    def think(self) -> str:
        # Focused state can be deliberative for complex goals or reactive for immediate threats related to focus.
        decision_mode = self.get_decision_mode() # Get the decision mode for this state
        return self.agent._think_default(decision_mode) # Pass the decision_mode to _think_default

    def act(self, action: str):
        self.agent._act_default(action)

    def get_decision_mode(self) -> str:
        """
        Focused state can be deliberative for long-term planning towards a goal,
        or reactive if the focus involves an immediate threat or opportunity.
        For simplicity, if focused on food (due to high hunger), it's reactive.
        Otherwise, it's deliberative for goal-seeking.
        """
        if self.agent.attention_focus == 'food' and self.agent.internal_state.hunger > 0.7:
            return 'reactive' # Urgent food need
        return 'deliberative' # General goal-oriented focus

