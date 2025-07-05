from core.emotion.base_emotion import EmotionStrategy

class BasicEmotionStrategy(EmotionStrategy):
    """
    Implements a basic strategy for updating an agent's emotional state.

    This strategy calculates emotions like joy, fear, frustration, and curiosity
    based on the agent's current perceptions (e.g., food availability, time of day)
    and internal physiological states (e.g., hunger, fatigue).
    """
    def update_emotions(self, perception: dict, internal_state):
        """
        Updates the agent's emotional states based on specific rules.

        This method calculates and sets values for 'joy', 'fear', 'frustration',
        and 'curiosity' in the agent's emotion_state.

        Args:
            perception (dict): A dictionary containing the agent's current sensory
                               inputs and environmental observations. Expected keys:
                               "food_available" (bool), "time_of_day" (str).
            internal_state: An object representing the agent's current internal
                            physiological states. Expected to have 'hunger' and
                            'fatigue' attributes (e.g., an InternalState instance).
        """
        # Joy: Increases when food is available, decreases with hunger.
        if perception.get("food_available"):
            # Agent feels more joy when food is present.
            self.emotion_state.set("joy", self.emotion_state.get("joy") + 0.05)
        else:
            # Joy decreases, proportional to current hunger level.
            self.emotion_state.set("joy", self.emotion_state.get("joy") - internal_state.hunger * 0.02)

        # Fear: Increases at night, decreases during the day.
        if perception.get("time_of_day") == "night":
            # Agent experiences more fear during nighttime.
            self.emotion_state.set("fear", self.emotion_state.get("fear") + 0.03)
        else:
            # Fear dissipates during daytime.
            self.emotion_state.set("fear", self.emotion_state.get("fear") - 0.02)

        # Frustration: Increases when hunger and fatigue levels are high.
        # Calculated as an average of hunger and fatigue, then scaled.
        frustration_level = (internal_state.hunger + internal_state.fatigue) / 2
        self.emotion_state.set("frustration", 0.6 * frustration_level)

        # Curiosity: High when basic physiological needs (hunger, fatigue) are low.
        if internal_state.hunger < 0.3 and internal_state.fatigue < 0.3:
            # Agent feels more curious when well-rested and fed.
            self.emotion_state.set("curiosity", 0.8)
        else:
            # Curiosity is lower when basic needs are more pressing.
            self.emotion_state.set("curiosity", 0.3)