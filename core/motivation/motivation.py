class MotivationEngine:
    def __init__(self, state):
        self.state = state

    def decide_action(self, perception=None, memory=None):
        hunger = self.state.hunger
        fatigue = self.state.fatigue

        # Default action
        action = "explore"

        # Memory-based decision
        food_available = perception.get("food_available", False)
        food_last_seen = memory.get("food_last_seen", None)
        step_now = memory.get("current_step", 0)
        food_recently_seen = (
            food_last_seen is not None and (step_now - food_last_seen) <= 3
        )

        if hunger > 0.6:
            if food_available or food_recently_seen:
                action = "seek_food"
            else:
                action = "explore"
        elif fatigue > 0.6:
            action = "rest"

        return action
