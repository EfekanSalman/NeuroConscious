"""Basic emotion strategy — rule-based emotion updates."""

from core.emotion.base_emotion import EmotionStrategy
from config.constants import (
    JOY_FOOD_BOOST,
    JOY_HUNGER_DECAY_FACTOR,
    FEAR_NIGHT_INCREASE,
    FEAR_DAY_DECREASE,
    CURIOSITY_HIGH,
    CURIOSITY_LOW,
    CURIOSITY_NEED_THRESHOLD,
)


class BasicEmotionStrategy(EmotionStrategy):
    """Simple rule-based emotion driver.

    - Joy rises with food availability, decays with hunger.
    - Fear increases at night.
    - Frustration tracks average unmet needs.
    - Curiosity spikes when all needs are satisfied.
    """

    def update_emotions(self, perception: dict, internal_state) -> None:
        # Joy
        if perception.get("food_available"):
            self.emotion_state.set("joy", self.emotion_state.get("joy") + JOY_FOOD_BOOST)
        else:
            self.emotion_state.set(
                "joy",
                self.emotion_state.get("joy") - internal_state.hunger * JOY_HUNGER_DECAY_FACTOR,
            )

        # Fear
        if perception.get("time_of_day") == "night":
            self.emotion_state.set("fear", self.emotion_state.get("fear") + FEAR_NIGHT_INCREASE)
        else:
            self.emotion_state.set("fear", self.emotion_state.get("fear") - FEAR_DAY_DECREASE)

        # Frustration — average of unmet needs
        frustration = (internal_state.hunger + internal_state.fatigue) / 2
        self.emotion_state.set("frustration", 0.6 * frustration)

        # Curiosity
        if (
            internal_state.hunger < CURIOSITY_NEED_THRESHOLD
            and internal_state.fatigue < CURIOSITY_NEED_THRESHOLD
        ):
            self.emotion_state.set("curiosity", CURIOSITY_HIGH)
        else:
            self.emotion_state.set("curiosity", CURIOSITY_LOW)
