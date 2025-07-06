from core.state import InternalState
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
        self.motivation_engine: BasicMotivationEngine = BasicMotivationEngine(agent=self)

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
        self._previous_state_snapshot = None  # This might be redundant if previous_physiological_states is used
        # Q-learning component for action selection based on states.
        self.q_learner: QTableLearner = QTableLearner(actions=["seek_food", "rest", "explore"])
        # Physiological state values (hunger, fatigue) before the last action, for reward calculation.
        self._previous_physiological_states = None
        # The last action performed by the agent.
        self._last_performed_action = None
        # The reward received from the last action.
        self.last_action_reward: float = 0.0

        # Manages the agent's emotional state.
        self.emotion_state: EmotionState = EmotionState()
        # Strategy for updating the agent's emotions.
        self.emotion_strategy: BasicEmotionStrategy = BasicEmotionStrategy(self.emotion_state)

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

        # 2. Update emotional state based on environmental perceptions and internal state.
        self.emotion_strategy.update_emotions(self.perception, self.internal_state)

        # 3. Store current state before acting (for learning)
        self._previous_physiological_states = (self.internal_state.hunger, self.internal_state.fatigue)

        # 4. Emotion-driven motivation engine decides preferred action
        # This part is currently not used for the final action decision as Q-learning overrides it.
        # It's kept here for potential future blending or debugging.
        _ = self.motivation_engine.decide_action(
            perception=self.perception,
            memory=self.short_term_memory,
            emotions=self.emotion_state
        )

        # 5. Q-learner selects an action. Currently, Q-learning overrides motivation.
        #    Future improvements could mix Q-learning with emotion preference later.
        selected_action = self.q_learner.choose_action(
            hunger=self.internal_state.hunger,
            fatigue=self.internal_state.fatigue
        )

        self._last_performed_action = selected_action
        return selected_action

    def act(self, action: str):
        """
        Executes the chosen action, updates the agent's internal state and Q-table,
        and records the experience in episodic memory.

        Args:
            action (str): The action decided by the agent's motivation engine.
                          Expected actions: "seek_food", "rest", "explore".
        """
        previous_internal_state = self.internal_state.snapshot()

        if action == "seek_food":
            if self.environment.food_available:
                self.internal_state.hunger = max(0.0, self.internal_state.hunger - 0.7)
                self.last_action_reward = 0.5  # Positive reward for successfully finding food.
                print(f"{self.name} successfully sought food! Hunger reduced.")
            else:
                self.internal_state.hunger = min(1.0, self.internal_state.hunger + 0.02)
                self.last_action_reward = -0.1  # Small penalty for unsuccessful attempt.
                print(f"{self.name} sought food but found none. Hunger increased slightly.")
            # Update memory about food presence
            if self.environment.food_available:
                self.short_term_memory["food_last_seen"] = self.current_time_step  # Use short_term_memory

        elif action == "rest":
            self.internal_state.fatigue = max(0.0, self.internal_state.fatigue - 0.6)
            self.last_action_reward = 0.4  # Positive reward for resting.
            print(f"{self.name} is resting. Fatigue reduced.")

        elif action == "explore":
            self.internal_state.hunger = min(1.0, self.internal_state.hunger + 0.05)
            self.internal_state.fatigue = min(1.0, self.internal_state.fatigue + 0.05)
            # Reward for exploration might be neutral or slight penalty initially.
            # This can be modified later based on discoveries during exploration.
            self.last_action_reward = -0.05
            print(f"{self.name} is exploring. Hunger/Fatigue increased slightly.")

        else:
            print(f"{self.name} attempted an unknown action: {action}")
            self.last_action_reward = -0.2  # Penalty for invalid action

        # Update the agent's internal state (hunger/fatigue naturally increase, mood recalculates)
        # Pass the current time step for delta_time (assuming 1.0 per step, or pass actual delta)
        self.internal_state.update(delta_time=1.0)  # Assume 1.0 delta_time for simplicity per step

        # Update the simple RewardLearner
        self.reward_learner.update(previous_internal_state, self.internal_state, action)

        # Update Q-table with the experience
        # Q-learner needs previous state, action taken, reward, and the new state
        self.q_learner.update(
            previous_internal_state.hunger, previous_internal_state.fatigue,
            action,
            self.last_action_reward,  # Use the reward calculated for this action
            self.internal_state.hunger, self.internal_state.fatigue
        )

        # Store the episode in episodic memory with emotional weighting
        self.episodic_memory.add(
            step=self.current_time_step,  # Use current_time_step
            perception=self.perception,  # Use the perception from the current step
            internal_state=previous_internal_state,  # Record state *before* action for episodic clarity
            action=action,
            emotions=self.emotion_state  # Pass the current emotion state for weighting
        )
        self.current_time_step += 1

    def log_status(self):
        """
        Prints the agent's current status, including internal state, emotions,
        and a sample of the Q-table.
        """
        print(f"{self.name} → {self.internal_state}")
        print("Emotions:", self.emotion_state)
        print("Q-table (sample):")
        print(self.q_learner)
        print("Episodic Memory (sample):")
        print(self.episodic_memory)
