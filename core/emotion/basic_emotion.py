from core.emotion.base_emotion import EmotionStrategy

class BasicEmotionStrategy(EmotionStrategy):
    def update_emotions(self, perception: dict, internal_state):
        # Joy goes up when food is available, down when hungry
        if perception.get("food_available"):
            self.emotion_state.set("joy", self.emotion_state.get("joy") + 0.05)
        else:
            self.emotion_state.set("joy", self.emotion_state.get("joy") - internal_state.hunger * 0.02)

        # Fear increases at night
        if perception.get("time_of_day") == "night":
            self.emotion_state.set("fear", self.emotion_state.get("fear") + 0.03)
        else:
            self.emotion_state.set("fear", self.emotion_state.get("fear") - 0.02)

        # Frustration increases if hunger and fatigue are high
        frustration = (internal_state.hunger + internal_state.fatigue) / 2
        self.emotion_state.set("frustration", 0.6 * frustration)

        # Curiosity = high when all needs are low
        if internal_state.hunger < 0.3 and internal_state.fatigue < 0.3:
            self.emotion_state.set("curiosity", 0.8)
        else:
            self.emotion_state.set("curiosity", 0.3)
