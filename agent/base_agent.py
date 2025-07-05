from core.state import InternalState
from core.motivation.motivation import MotivationEngine
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.reward_learner import RewardLearner
from core.learning.q_table_learner import QTableLearner
from core.emotion.emotion_state import EmotionState
from core.emotion.basic_emotion import BasicEmotionStrategy
from core.motivation.basic_motivation import BasicMotivationEngine


class Agent:
    """
    Represents an autonomous agent within the NeuroConscious simulation.

    This class encapsulates the agent's internal state, motivation, perception,
    memory, and learning capabilities, allowing it to interact with an
    environment and make decisions.
    """
    def __init__(self, name: str, mood_strategy: MoodStrategy):
        """
        Initializes a new Agent instance.

        Args:
            name (str): The unique name of the agent.
            mood_strategy (MoodStrategy): The strategy used to calculate the agent's mood.
        """
        self.name: str = name
        # Manages physiological states like hunger and fatigue, and mood.
        self.internal_state: InternalState = InternalState(mood_strategy)
        # Note: MotivationEngine is initialized twice. Consider if this is intentional.
        # The second initialization with BasicMotivationEngine will override the first.
        self.motivation_engine: MotivationEngine = MotivationEngine(self.internal_state)

        # Perceptual data about the environment.
        self.perception: dict = {
            "food_available": False,
            "time_of_day": "day"
        }
        # Short-term and context-specific memory.
        self.short_term_memory: dict = {
            "food_last_seen": None
        }
        # Reference to the simulation environment the agent interacts with.
        self.environment = None
        # Current simulation time step.
        self.current_time_step: int = 0

        # Long-term memory for past experiences.
        self.episodic_memory: EpisodicMemory = EpisodicMemory(capacity=5)
        # Learner for generic reward-based learning (e.g., policy gradients - though not fully implemented here).
        self.reward_learner: RewardLearner = RewardLearner()
        # Snapshot of the agent's state before an action, used for learning.
        self._previous_state_snapshot = None
        # Q-learning component for action selection based on states.
        self.q_learner: QTableLearner = QTableLearner(actions=["seek_food", "rest", "explore"])
        # Physiological state values (hunger, fatigue) before the last action, for reward calculation.
        self._previous_physiological_states = None
        # The last action performed by the agent.
        self._last_performed_action = None

        # Manages the agent's emotional state.
        self.emotion_state: EmotionState = EmotionState()
        # Strategy for updating the agent's emotions.
        self.emotion_strategy: BasicEmotionStrategy = BasicEmotionStrategy(self.emotion_state)
        # Basic motivation engine that drives action selection based on internal needs.
        # This re-initializes `self.motivation_engine` from above.
        self.motivation_engine = BasicMotivationEngine(agent=self)


    def set_environment(self, env):
        """
        Sets the simulation environment for the agent.

        Args:
            env: An instance of the simulation environment (e.g., World class).
        """
        self.environment = env

    def sense(self):
        """
        Updates the agent's perceptions based on the current environment state.

        This method reads information like food availability and time of day
        from the linked environment and updates the agent's internal perception
        and short-term memory.
        """
        if self.environment:
            self.perception["food_available"] = self.environment.food_available
            self.perception["time_of_day"] = self.environment.time_of_day
            self.current_time_step = self.environment.time_step

            # Records the last time food was perceived as available.
            if self.environment.food_available:
                self.short_term_memory["food_last_seen"] = self.current_time_step

    def think(self) -> str:
        """
        Determines the agent's next action based on its internal state, emotions,
        and learning algorithms.

        The thinking process involves updating physiological states, emotional states,
        and then using a combination of motivation and Q-learning to select an action.

        Returns:
            str: The chosen action (e.g., "seek_food", "rest", "explore").
        """
        # 1. Update physiological state (e.g., hunger, fatigue increases over time).
        # Hunger/fatigue increases faster at night.
        time_delta_factor = 1.5 if self.perception["time_of_day"] == "night" else 1.0
        self.internal_state.update(delta_time=time_delta_factor)

        # 2. Update emotional state based on environmental perceptions and internal needs.
        self.emotion_strategy.update_emotions(self.perception, self.internal_state)

        # 3. Store current physiological state before deciding on an action, for later reward calculation.
        self._previous_physiological_states = (self.internal_state.hunger, self.internal_state.fatigue)

        # 4. The motivation engine proposes an action based on current needs and emotions.
        motivation_driven_action = self.motivation_engine.decide_action(
            perception=self.perception,
            memory=self.short_term_memory,
            emotions=self.emotion_state
        )

        # 5. Q-learner selects an action. Currently, Q-learning overrides motivation.
        #    Future improvements could blend these two decision mechanisms.
        selected_action = self.q_learner.choose_action(
            hunger=self.internal_state.hunger,
            fatigue=self.internal_state.fatigue
        )

        self._last_performed_action = selected_action
        return selected_action

    def act(self, action: str):
        """
        Executes the chosen action and updates the agent's internal state
        and Q-learning table based on the outcome.

        Args:
            action (str): The action to be performed (e.g., "seek_food", "rest", "explore").
        """
        # Perform the action and apply its immediate effects on internal state.
        if action == "seek_food" and self.perception["food_available"]:
            print(f"{self.name} eats food.")
            self.internal_state.hunger = max(0.0, self.internal_state.hunger - 0.4)
        elif action == "seek_food":
            print(f"{self.name} searches for food but finds none.")
        elif action == "rest":
            print(f"{self.name} takes a rest.")
            self.internal_state.fatigue = max(0.0, self.internal_state.fatigue - 0.3)
        elif action == "explore":
            print(f"{self.name} is exploring...")

        # Learn from the action: Update Q-table using the reward.
        if self._previous_physiological_states and self._last_performed_action:
            prev_hunger, prev_fatigue = self._previous_physiological_states
            current_hunger, current_fatigue = self.internal_state.hunger, self.internal_state.fatigue

            # Define a simple reward signal based on action outcomes.
            reward_value = 0.0
            if self._last_performed_action == "seek_food":
                # Reward for reducing hunger.
                reward_value = max(0, prev_hunger - current_hunger)
            elif self._last_performed_action == "rest":
                # Reward for reducing fatigue.
                reward_value = max(0, prev_fatigue - current_fatigue)
            elif self._last_performed_action == "explore":
                # Small penalty for exploration, as it consumes time/energy without direct gain.
                reward_value = -0.01

            self.q_learner.update(
                prev_hunger = prev_hunger,
                prev_fatigue = prev_fatigue,
                action = self._last_performed_action,
                reward = reward_value,
                next_hunger = current_hunger,
                next_fatigue = current_fatigue
            )

        # Store the current event and state in the episodic memory for later recall/analysis.
        self.episodic_memory.add(
            step = self.current_time_step,
            perception = self.perception,
            state = self.internal_state,
            action = action
        )

    def log_status(self):
        """
        Prints the agent's current status, including internal state, emotions,
        and a sample of the Q-table.
        """
        print(f"{self.name} → {self.internal_state}")
        print("Emotions:", self.emotion_state)
        print("Q-table (sample):")
        print(self.q_learner)