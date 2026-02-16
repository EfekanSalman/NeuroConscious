"""Emotion-driven motivation engine with priority-based decision making."""

import random

from utils.logger import get_logger
from config.constants import (
    FRUSTRATION_OVERRIDE_THRESHOLD,
    FEAR_INHIBIT_THRESHOLD,
    FEAR_SECONDARY_THRESHOLD,
    CURIOSITY_EXPLORE_THRESHOLD,
    CURIOSITY_NEED_THRESHOLD,
    JOY_EXPLORE_THRESHOLD,
    JOY_EXPLORE_PROBABILITY,
    NEED_ACTION_THRESHOLD,
)

logger = get_logger(__name__)


class BasicMotivationEngine:
    """Priority-cascading motivation system influenced by emotions.

    Decision cascade:
        1. Frustration emergency → satisfy highest need
        2. Fear inhibition → avoid exploration
        3. Curiosity + low needs → explore
        4. Joy + random → explore
        5. Physical need fallback
        6. Default: explore
    """

    def __init__(self, agent):
        self.agent = agent

    def decide_action(self, perception: dict, memory: dict, emotions) -> str:
        """Return the emotion-driven preferred action.

        Args:
            perception: Current sensory data.
            memory: Agent's memory store.
            emotions: EmotionState instance.

        Returns:
            Action string: "seek_food", "rest", or "explore".
        """
        hunger = self.agent.state.hunger
        fatigue = self.agent.state.fatigue

        joy = emotions.get("joy")
        fear = emotions.get("fear")
        frustration = emotions.get("frustration")
        curiosity = emotions.get("curiosity")

        logger.debug(
            "Motivation — H:%.2f F:%.2f | Joy:%.2f Fear:%.2f Frust:%.2f Cur:%.2f",
            hunger, fatigue, joy, fear, frustration, curiosity,
        )

        # 1. Frustration emergency
        if frustration > FRUSTRATION_OVERRIDE_THRESHOLD:
            return "seek_food" if hunger > fatigue else "rest"

        # 2. Fear inhibits exploration
        if fear > FEAR_INHIBIT_THRESHOLD:
            if fatigue > FEAR_SECONDARY_THRESHOLD:
                return "rest"
            return "seek_food" if hunger > FEAR_SECONDARY_THRESHOLD else "rest"

        # 3. Curiosity promotes exploration
        if (
            curiosity > CURIOSITY_EXPLORE_THRESHOLD
            and hunger < CURIOSITY_NEED_THRESHOLD
            and fatigue < CURIOSITY_NEED_THRESHOLD
        ):
            return "explore"

        # 4. Joy encourages non-urgent exploration
        if joy > JOY_EXPLORE_THRESHOLD and random.random() < JOY_EXPLORE_PROBABILITY:
            return "explore"

        # 5. Physical need fallback
        if hunger > NEED_ACTION_THRESHOLD:
            return "seek_food"
        if fatigue > NEED_ACTION_THRESHOLD:
            return "rest"

        # 6. Default
        return "explore"
