from core.state import InternalState
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.reward_learner import RewardLearner
from core.learning.q_table_learner import QTableLearner
from core.emotion.emotion_state import EmotionState
from core.emotion.basic_emotion import BasicEmotionStrategy
from core.motivation.basic_motivation import BasicMotivationEngine
from typing import List, Tuple  # For type hinting grid coordinates


class Agent:
    """
    Represents an autonomous agent within the NeuroConscious simulation.

    This class encapsulates the agent's internal state, motivation, perception,
    memory, and learning capabilities, allowing it to interact with an
    environment and make decisions. It now operates within a grid-based world.
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

        # Motivation engine that drives action selection based on internal needs and emotions.
        self.motivation_engine: BasicMotivationEngine = BasicMotivationEngine(agent=self)

        # Perceptual data about the environment, including local grid view.
        self.perception: dict = {
            "food_available_global": False,  # Global flag from World
            "time_of_day": "day",
            "local_grid_view": [],  # Agent's view of its immediate surroundings
            "food_in_sight": False  # Whether food is in agent's local view
        }
        # Short-term and context-specific memory.
        self.short_term_memory: dict = {
            "food_last_seen": None
        }
        # Reference to the simulation environment the agent interacts with.
        self.environment = None
        # Current simulation time step.
        self.current_time_step: int = 0

        # Agent's position on the grid (row, column)
        self.pos_x: int = 0
        self.pos_y: int = 0

        # Long-term memory for past experiences.
        self.episodic_memory: EpisodicMemory = EpisodicMemory(capacity=5)
        # Learner for generic reward-based learning (e.g., policy gradients - though not fully implemented here).
        self.reward_learner: RewardLearner = RewardLearner()
        # Snapshot of the agent's state before an action, used for learning.
        self._previous_state_snapshot = None
        # Q-learning component for action selection based on states.
        self.q_learner: QTableLearner = QTableLearner(actions=["seek_food", "rest", "explore",
                                                               "move_up", "move_down", "move_left", "move_right"])
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
        Updates the agent's perceptions based on the current environment state and local grid view.

        This method reads global information (food availability, time of day) from the World
        and also scans the immediate surroundings (e.g., 1-cell radius) on the grid for items like food.
        """
        if self.environment:
            # Global perceptions
            self.perception["food_available_global"] = self.environment.food_available
            self.perception["time_of_day"] = self.environment.time_of_day
            self.current_time_step = self.environment.time_step

            # Local grid perception (e.g., 1-cell radius around the agent)
            self.perception["local_grid_view"] = self._get_local_grid_view(radius=1)

            # Check for food in local view
            self.perception["food_in_sight"] = False
            for row in self.perception["local_grid_view"]:
                if 'food' in row:
                    self.perception["food_in_sight"] = True
                    break

            # Records the last time food was perceived as available (either globally or locally).
            if self.perception["food_available_global"] or self.perception["food_in_sight"]:
                self.short_term_memory["food_last_seen"] = self.current_time_step

    def _get_local_grid_view(self, radius: int = 1) -> List[List[str]]:
        """
        Retrieves a square view of the grid centered around the agent's position.

        Args:
            radius (int): The radius of the square view (e.g., radius=1 means a 3x3 view).

        Returns:
            List[List[str]]: A 2D list representing the agent's local view of the grid.
                             Cells outside the grid boundaries will be represented as 'boundary'.
        """
        view = []
        grid_size = len(self.environment.grid)  # Assuming square grid

        for r_offset in range(-radius, radius + 1):
            row_view = []
            for c_offset in range(-radius, radius + 1):
                view_row, view_col = self.pos_x + r_offset, self.pos_y + c_offset
                if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                    row_view.append(self.environment.grid[view_row][view_col])
                else:
                    row_view.append('boundary')  # Mark cells outside the grid
            view.append(row_view)
        return view

    def think(self) -> str:
        """
        Determines the agent's next action based on its internal state, emotions,
        and learning algorithms.

        The thinking process involves updating physiological states, emotional states,
        and then using a combination of motivation and Q-learning to select an action.

        Returns:
            str: The chosen action (e.g., "seek_food", "rest", "explore", "move_up", etc.).
        """
        # 1. Update physiological state (e.g., hunger, fatigue increases over time).
        time_delta_factor = 1.5 if self.perception["time_of_day"] == "night" else 1.0
        self.internal_state.update(delta_time=time_delta_factor)

        # 2. Update emotional state based on environmental perceptions and internal state.
        self.emotion_strategy.update_emotions(self.perception, self.internal_state)

        # 3. Store current state before acting (for learning)
        self._previous_physiological_states = (self.internal_state.hunger, self.internal_state.fatigue)

        # 4. Emotion-driven motivation engine decides preferred action
        # This part is currently not used for the final action decision as Q-learning overrides it.
        _ = self.motivation_engine.decide_action(
            perception=self.perception,
            memory=self.short_term_memory,
            emotions=self.emotion_state
        )

        # 5. Q-learner selects an action.
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
        This method now includes movement logic for grid-based actions.

        Args:
            action (str): The action decided by the agent's motivation engine.
                          Expected actions: "seek_food", "rest", "explore",
                          "move_up", "move_down", "move_left", "move_right".
        """
        # We get a copy of the agent's internal state before the action.
        previous_internal_state = self.internal_state.snapshot()
        original_pos_x, original_pos_y = self.pos_x, self.pos_y  # Store original position

        #  Perform the selected action and implement its effects
        if action == "seek_food":
            # Check if there's food at the agent's current location
            if self.environment.grid[self.pos_x][self.pos_y] == 'food':
                self.internal_state.hunger = max(0.0, self.internal_state.hunger - 0.7)
                self.last_action_reward = 0.5  # Positive reward for successfully finding food.
                self.environment.grid[self.pos_x][self.pos_y] = 'empty'  # Consume food
                print(f"{self.name} successfully ate food at ({self.pos_x},{self.pos_y})! Hunger reduced.")
            else:
                self.internal_state.hunger = min(1.0, self.internal_state.hunger + 0.02)
                self.last_action_reward = -0.1  # Small penalty for unsuccessful attempt.
                print(
                    f"{self.name} sought food but found none at ({self.pos_x},{self.pos_y}). Hunger increased slightly.")
            # Update memory about food presence (if food was globally available or locally seen)
            if self.perception["food_available_global"] or self.perception["food_in_sight"]:
                self.short_term_memory["food_last_seen"] = self.current_time_step

        elif action == "rest":
            self.internal_state.fatigue = max(0.0, self.internal_state.fatigue - 0.6)
            self.last_action_reward = 0.4  # Positive reward for resting.
            print(f"{self.name} is resting at ({self.pos_x},{self.pos_y}). Fatigue reduced.")

        elif action == "explore":
            self.internal_state.hunger = min(1.0, self.internal_state.hunger + 0.05)
            self.internal_state.fatigue = min(1.0, self.internal_state.fatigue + 0.05)
            self.last_action_reward = -0.05
            print(f"{self.name} is exploring at ({self.pos_x},{self.pos_y}). Hunger/Fatigue increased slightly.")

        # Movement Actions
        elif action.startswith("move_"):
            new_pos_x, new_pos_y = self.pos_x, self.pos_y
            if action == "move_up":
                new_pos_x -= 1
            elif action == "move_down":
                new_pos_x += 1
            elif action == "move_left":
                new_pos_y -= 1
            elif action == "move_right":
                new_pos_y += 1

            # Check if new position is within grid boundaries
            grid_size = len(self.environment.grid)
            if 0 <= new_pos_x < grid_size and 0 <= new_pos_y < grid_size:
                # For simplicity, agents can move into any cell (empty or food).
                self.pos_x, self.pos_y = new_pos_x, new_pos_y
                self.last_action_reward = -0.01  # Small cost for movement
                print(f"{self.name} moved {action.replace('move_', '')} to ({self.pos_x},{self.pos_y}).")
            else:
                self.last_action_reward = -0.1  # Penalty for trying to move out of bounds
                print(
                    f"{self.name} tried to move {action.replace('move_', '')} out of bounds from ({original_pos_x},{original_pos_y}).")

        else:
            print(f"{self.name} attempted an unknown action: {action}")
            self.last_action_reward = -0.2  # Penalty for invalid action

        # Update the agent's internal state (hunger/fatigue naturally increase, mood recalculates)
        self.internal_state.update(delta_time=1.0)

        # Update the simple RewardLearner
        self.reward_learner.update(previous_internal_state, self.internal_state, action)

        # Update Q-table with the experience
        self.q_learner.update(
            previous_internal_state.hunger, previous_internal_state.fatigue,
            action,
            self.last_action_reward,
            self.internal_state.hunger, self.internal_state.fatigue
        )

        # Store the episode in episodic memory with emotional weighting
        self.episodic_memory.add(
            step=self.current_time_step,
            perception=self.perception,
            internal_state=previous_internal_state,  # Record state *before* action for episodic clarity
            action=action,
            emotions=self.emotion_state
        )

        # Increment agent's internal step counter (for memory and other time-based checks)
        self.current_time_step += 1

    def log_status(self):
        """
        Prints the agent's current status, including internal state, emotions,
        Q-table sample, and episodic memory sample.
        """
        print(f"{self.name} at ({self.pos_x},{self.pos_y}) → {self.internal_state}")
        print("Emotions:", self.emotion_state)
        print("Q-table (sample):")
        print(self.q_learner)
        print("Episodic Memory (sample):")
        print(self.episodic_memory)
        print("Local Perception (3x3 view):")
        # Print local grid view for debugging
        if self.perception["local_grid_view"]:
            for row in self.perception["local_grid_view"]:
                print(" ".join(row))
        else:
            print("No local view available (sense() not called or environment not set).")

