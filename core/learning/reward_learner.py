from collections import defaultdict

class RewardLearner:
    """
    A simple learning mechanism that tracks and accumulates rewards for specific actions.

    This learner helps the agent develop a basic preference for actions that have
    historically led to positive outcomes, particularly in satisfying immediate needs
    like hunger and fatigue.
    """
    def __init__(self):
        """
        Initializes the RewardLearner with an empty record of action rewards.

        The action_rewards dictionary will store the cumulative reward associated
        with each specific action taken by the agent.
        """
        # Stores cumulative reward for each action. defaultdict ensures new actions start with 0.0 reward.
        self.action_rewards: defaultdict[str, float] = defaultdict(float)

    def update(self, previous_state, current_state, action: str):
        """
        Updates the cumulative reward for a given action based on its outcome.

        This method calculates a reward based on how well the action helped in
        reducing hunger or fatigue, and then adds this reward to the action's
        total.

        Args:
            previous_state: The agent's internal state (e.g., InternalState instance)
                            before the 'action' was performed. Expected to have 'hunger'
                            and 'fatigue' attributes.
            current_state: The agent's internal state (e.g., InternalState instance)
                           after the 'action' was performed. Expected to have 'hunger'
                           and 'fatigue' attributes.
            action (str): The name of the action that was just performed (e.g., "seek_food", "rest").
        """
        reward_value: float = 0.0

        if action == "seek_food":
            # Reward is given if hunger was successfully reduced.
            hunger_change = previous_state.hunger - current_state.hunger
            reward_value = max(0, hunger_change)  # Only positive change (reduction) is rewarded.
        elif action == "rest":
            # Reward is given if fatigue was successfully reduced.
            fatigue_change = previous_state.fatigue - current_state.fatigue
            reward_value = max(0, fatigue_change) # Only positive change (reduction) is rewarded.

        # Accumulate the calculated reward for the specific action.
        self.action_rewards[action] += reward_value

    def get_best_action(self, hunger: float, fatigue: float) -> str | None:
        """
        Suggests an action based on current needs and learned rewards.

        This method provides a simple behavioral bias: if a need (hunger or fatigue)
        is high, it recommends the action (seek_food or rest) that has historically
        provided more reward for addressing that need.

        Args:
            hunger (float): The agent's current hunger level.
            fatigue (float): The agent's current fatigue level.

        Returns:
            str | None: The suggested action ("seek_food" or "rest") if a clear
                        preference exists and a need is high; otherwise, None.
        """
        # If hunger is high, check if 'seek_food' has been more rewarding than 'rest'.
        if hunger > 0.6 and self.action_rewards["seek_food"] > self.action_rewards["rest"]:
            return "seek_food"
        # If fatigue is high, check if 'rest' has been more rewarding than 'seek_food'.
        if fatigue > 0.6 and self.action_rewards["rest"] > self.action_rewards["seek_food"]:
            return "rest"
        return None  # No strong preference based on learned rewards or needs.

    def __str__(self) -> str:
        """
        Provides a string representation of the accumulated action rewards.

        This is useful for debugging and understanding which actions have been
        most beneficial to the agent over time.

        Returns:
            str: A string showing the dictionary of action names and their cumulative rewards.
        """
        return str(dict(self.action_rewards))