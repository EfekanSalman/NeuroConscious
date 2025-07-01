class MotivationEngine:
    def __init__(self, internal_state):
        self.state = internal_state

    def decide_action(self) -> str:
        """According to the internal situation the agent decides on action."""
        if self.state.hunger > 0.7:
            return "seek_food"
        elif self.state.fatigue > 0.7:
            return "rest"
        return "explore"