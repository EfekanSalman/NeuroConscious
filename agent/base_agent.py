"""
Agent — Core autonomous agent with emotion-influenced Q-learning.

The agent perceives the world, processes emotions, and makes decisions
using a hybrid system: emotion-driven motivation weighted against
tabular Q-learning exploration/exploitation.
"""

from core.state import InternalState
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.q_table_learner import QTableLearner
from core.emotion.emotion_state import EmotionState
from core.emotion.basic_emotion import BasicEmotionStrategy
from core.motivation.basic_motivation import BasicMotivationEngine
from utils.logger import get_logger
from config.constants import (
    FOOD_HUNGER_REDUCTION,
    REST_FATIGUE_REDUCTION,
    EXPLORE_PENALTY,
    NIGHT_TIME_MULTIPLIER,
    EMOTION_WEIGHT,
    EPISODIC_MEMORY_CAPACITY,
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON,
)

logger = get_logger(__name__)

ACTIONS = ["seek_food", "rest", "explore"]


class Agent:
    """Autonomous agent with physiological needs, emotions, and learning.

    Decision pipeline per tick:
        1. sense()  — read environment into perception
        2. think()  — update state → emotions → hybrid decision
        3. act()    — apply action effects, update Q-table, store memory
    """

    def __init__(self, name: str, mood_strategy: MoodStrategy):
        self.name = name
        self.state = InternalState(mood_strategy)

        # Perception & simple memory
        self.perception: dict = {
            "food_available": False,
            "time_of_day": "day",
        }
        self.memory: dict = {
            "food_last_seen": None,
            "current_step": 0,
        }

        self.environment = None
        self.current_step: int = 0

        # Subsystems
        self.episodic_memory = EpisodicMemory(capacity=EPISODIC_MEMORY_CAPACITY)
        self.q_learner = QTableLearner(
            actions=ACTIONS,
            alpha=Q_LEARNING_ALPHA,
            gamma=Q_LEARNING_GAMMA,
            epsilon=Q_LEARNING_EPSILON,
        )
        self.emotion_state = EmotionState()
        self.emotion = BasicEmotionStrategy(self.emotion_state)
        self.motivation = BasicMotivationEngine(agent=self)

        # Learning bookkeeping
        self._prev_state_vals: tuple | None = None
        self._last_action: str | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    def set_environment(self, env) -> None:
        """Bind agent to a World instance."""
        self.environment = env

    def sense(self) -> None:
        """Read environment state into internal perception."""
        if not self.environment:
            return

        self.perception["food_available"] = self.environment.food_available
        self.perception["time_of_day"] = self.environment.time_of_day
        self.current_step = self.environment.time_step

        # Update memory
        if self.environment.food_available:
            self.memory["food_last_seen"] = self.current_step
        self.memory["current_step"] = self.current_step

    def think(self) -> str:
        """Run the full decision pipeline and return an action string.

        Pipeline:
            1. Update physiological state (hunger, fatigue).
            2. Update emotional state.
            3. Get emotion-driven action from motivation engine.
            4. Get exploitation/exploration action from Q-learner.
            5. Blend via hybrid weighting.
        """
        # 1. Physiological update
        delta = NIGHT_TIME_MULTIPLIER if self.perception["time_of_day"] == "night" else 1.0
        self.state.update(delta_time=delta)

        # 2. Emotional update
        self.emotion.update_emotions(self.perception, self.state)

        # 3. Snapshot state before acting (for Q-learning reward)
        self._prev_state_vals = (self.state.hunger, self.state.fatigue)

        # 4. Emotion-driven preferred action
        emotion_action = self.motivation.decide_action(
            perception=self.perception,
            memory=self.memory,
            emotions=self.emotion_state,
        )

        # 5. Q-learning action
        q_action = self.q_learner.choose_action(
            hunger=self.state.hunger,
            fatigue=self.state.fatigue,
        )

        # 6. Hybrid blend — emotion overrides Q when weight threshold met
        import random
        action = emotion_action if random.random() < EMOTION_WEIGHT else q_action

        self._last_action = action
        logger.debug("Step %d: emotion→%s  Q→%s  chosen→%s", self.current_step, emotion_action, q_action, action)
        return action

    def act(self, action: str) -> None:
        """Execute the chosen action and learn from the outcome."""
        # Apply action effects
        if action == "seek_food" and self.perception["food_available"]:
            logger.info("%s eats food.", self.name)
            self.state.hunger = max(0.0, self.state.hunger - FOOD_HUNGER_REDUCTION)
        elif action == "seek_food":
            logger.info("%s searches for food but finds none.", self.name)
        elif action == "rest":
            logger.info("%s takes a rest.", self.name)
            self.state.fatigue = max(0.0, self.state.fatigue - REST_FATIGUE_REDUCTION)
        elif action == "explore":
            logger.info("%s is exploring...", self.name)

        # Q-learning update
        if self._prev_state_vals and self._last_action:
            prev_h, prev_f = self._prev_state_vals
            next_h, next_f = self.state.hunger, self.state.fatigue

            reward = 0.0
            if self._last_action == "seek_food":
                reward = max(0, prev_h - next_h)
            elif self._last_action == "rest":
                reward = max(0, prev_f - next_f)
            elif self._last_action == "explore":
                reward = EXPLORE_PENALTY

            self.q_learner.update(
                prev_hunger=prev_h,
                prev_fatigue=prev_f,
                action=self._last_action,
                reward=reward,
                next_hunger=next_h,
                next_fatigue=next_f,
            )

        # Episodic memory
        self.episodic_memory.add(
            step=self.current_step,
            perception=self.perception,
            state=self.state,
            action=action,
        )

    def log_status(self) -> None:
        """Print a human-readable status summary."""
        print(f"{self.name} → {self.state}")
        print(f"  Emotions: {self.emotion_state}")
        print(f"  Q-table (sample): {self.q_learner}")