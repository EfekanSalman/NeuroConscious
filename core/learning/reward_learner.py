class RewardLearner:
    """
    A simple learner responsible for processing rewards based on state transitions.

    In a more complex system, this might involve learning a reward function
    or shaping rewards for other learning algorithms. This version can adjust
    rewards based on the agent's current mood.
    """
    def __init__(self):
        """
        Initializes the RewardLearner.
        """
        pass # Currently, no specific parameters are needed for initialization.

    def update(self, previous_state, current_state, action: str, raw_reward: float, mood_value: float):
        """
        Processes and potentially adjusts a raw reward based on the agent's mood.

        This method simulates how an agent's emotional state might influence its
        perception of rewards, making learning more emotionally nuanced.

        Args:
            previous_state: The agent's internal state before the action.
            current_state: The agent's internal state after the action.
            action (str): The action performed.
            raw_reward (float): The immediate reward received from the environment.
            mood_value (float): The current numerical mood value of the agent (0.0 to 1.0).
                                0.0 typically means very negative mood, 1.0 means very positive.
        """
        adjusted_reward = raw_reward

        # Simple mood-based reward adjustment:
        # Positive mood (mood_value > 0.5) might amplify positive rewards or dampen negative ones.
        # Negative mood (mood_value < 0.5) might dampen positive rewards or amplify negative ones.

        if mood_value > 0.7: # Very positive mood (e.g., happy)
            if raw_reward > 0:
                adjusted_reward *= 1.2 # Amplify positive rewards by 20%
                print(f"[REWARD LEARNER] Positive mood ({mood_value:.2f}) amplified reward: {raw_reward:.2f} -> {adjusted_reward:.2f}")
            elif raw_reward < 0:
                adjusted_reward *= 0.8 # Dampen negative rewards by 20%
                print(f"[REWARD LEARNER] Positive mood ({mood_value:.2f}) dampened penalty: {raw_reward:.2f} -> {adjusted_reward:.2f}")
        elif mood_value < 0.3: # Very negative mood (e.g., sad, frustrated)
            if raw_reward > 0:
                adjusted_reward *= 0.7 # Dampen positive rewards by 30%
                print(f"[REWARD LEARNER] Negative mood ({mood_value:.2f}) dampened reward: {raw_reward:.2f} -> {adjusted_reward:.2f}")
            elif raw_reward < 0:
                adjusted_reward *= 1.3 # Amplify negative rewards (penalties) by 30%
                print(f"[REWARD LEARNER] Negative mood ({mood_value:.2f}) amplified penalty: {raw_reward:.2f} -> {adjusted_reward:.2f}")
        else: # Neutral mood, no significant adjustment
            pass # Reward remains raw_reward

        return adjusted_reward


