from collections import defaultdict

class RewardLearner:
    def __init__(self):
        # Reward values for actions (simple memory)
        self.action_rewards = defaultdict(float)

    def update(self, previous_state, current_state, action):
        """Assign reward based on what the action achieved."""
        reward = 0.0

        if action == "seek_food":
            hunger_change = previous_state.hunger - current_state.hunger
            reward = max(0, hunger_change)  # only reward reduction
        elif action == "rest":
            fatigue_change = previous_state.fatigue - current_state.fatigue
            reward = max(0, fatigue_change)

        self.action_rewards[action] += reward

    def get_best_action(self, hunger, fatigue):
        """Bias action based on learned rewards."""
        if hunger > 0.6 and self.action_rewards["seek_food"] > self.action_rewards["rest"]:
            return "seek_food"
        if fatigue > 0.6 and self.action_rewards["rest"] > self.action_rewards["seek_food"]:
            return "rest"
        return None  # No preference

    def __str__(self):
        return str(dict(self.action_rewards))
