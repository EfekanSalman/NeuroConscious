# agent/base_agent.py

# Ensure all necessary imports are at the top of your file:
import random
from collections import defaultdict, deque  # Import deque for working memory
from core.state import InternalState
from core.motivation.motivation import MotivationEngine  # Or BasicMotivationEngine if you're using that directly
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.learning.reward_learner import RewardLearner
from core.learning.dqn_learner import DQNLearner  # Changed: Import DQNLearner
from core.emotion.emotion_state import EmotionState
from core.emotion.basic_emotion import BasicEmotionStrategy
from core.motivation.basic_motivation import BasicMotivationEngine  # If you're using this one
from typing import List, Tuple, Dict


class Agent:
    """
    Represents an autonomous agent within the NeuroConscious simulation.

    This class encapsulates the agent's internal state, motivation, perception,
    memory, and learning capabilities, allowing it to interact with an
    environment and make decisions. It now operates within a grid-based world
    and can perceive other agents, with its episodic memory beginning to influence behavior,
    and can also perceive and react to weather conditions. This version also
    introduces sensory noise, making perception imperfect, and incorporates
    curiosity-driven exploration, along with a basic goal system,
    integrates mood-based reward adjustment, a working memory buffer,
    a basic attention system, and uses a Deep Q-Network (DQN) for learning.
    """

    def __init__(self, name: str, mood_strategy: MoodStrategy, perception_accuracy: float = 0.95):
        """
        Initializes a new Agent instance.

        Args:
            name (str): The unique name of the agent.
            mood_strategy (MoodStrategy): The strategy used to calculate the agent's mood.
            perception_accuracy (float, optional): The probability (0.0 to 1.0) that the agent
                                                  will correctly perceive an item in its view.
                                                  Defaults to 0.95 (95% accurate).
        """
        self.name: str = name
        # Manages physiological states like hunger and fatigue, and mood.
        self.internal_state: InternalState = InternalState(mood_strategy)

        # Motivation engine that drives action selection based on internal needs and emotions.
        self.motivation_engine: BasicMotivationEngine = BasicMotivationEngine(agent=self)

        # Perceptual data about the environment, including local grid view, other agents, and weather.
        self.perception: dict = {
            "food_available_global": False,  # Global flag from World
            "time_of_day": "day",
            "current_weather": "sunny",  # Current weather condition
            "local_grid_view": [],  # Agent's view of its immediate surroundings
            "food_in_sight": False,  # Whether food is in agent's local view
            "other_agents_in_sight": []  # List of (agent_name, pos_x, pos_y) of other agents in local view
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

        # Changed: Use DQNLearner instead of QTableLearner
        # State size for DQN is 2 (hunger, fatigue)
        self.q_learner: DQNLearner = DQNLearner(
            actions=["seek_food", "rest", "explore", "move_up", "move_down", "move_left", "move_right"],
            state_size=2  # Hunger and Fatigue
        )

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

        # Perception accuracy for sensory noise
        self.perception_accuracy: float = perception_accuracy

        # Track visited states for curiosity
        self.visited_states: Dict[str, int] = defaultdict(int)  # Counts how many times a state has been visited

        # Basic Goal System
        # Goals can be represented as dictionaries: {"type": "reach_location", "target_x": 5, "target_y": 5, "priority": 0.8, "completed": False}
        # Or {"type": "maintain_hunger", "threshold": 0.3, "duration": 10, "current_duration": 0, "priority": 0.6}
        self.active_goals: List[Dict] = []
        self._initialize_goals()  # Add some initial goals

        # Working Memory Buffer
        # Stores recent important perceptions or thoughts for short-term recall.
        # Max length of 5 items, oldest items are discarded when new ones are added.
        self.working_memory_buffer: deque = deque(maxlen=5)

        # Attention System
        # What the agent is currently focusing its attention on.
        # Can be 'food', 'other_agents', 'curiosity', 'goal', or None (default/diffuse attention).
        self.attention_focus: str = None

    def _initialize_goals(self):
        """
        Initializes a set of default goals for the agent.
        This can be expanded later to dynamically generate goals.
        """
        # Example Goal 1: Reach a specific location (e.g., center of the map)
        self.active_goals.append({
            "type": "reach_location",
            "target_x": 5,
            "target_y": 5,
            "priority": 0.7,  # Higher priority means more influential
            "completed": False,  # Ensure 'completed' key is present
            "name": "Reach Center"
        })
        # Example Goal 2: Maintain low hunger for a period
        self.active_goals.append({
            "type": "maintain_hunger_low",
            "threshold": 0.3,  # Keep hunger below this
            "duration_steps": 20,  # For 20 consecutive steps
            "current_duration": 0,
            "priority": 0.6,
            "name": "Stay Fed",
            "completed": False  # Ensure 'completed' key is present
        })
        print(f"{self.name} initialized with goals: {[goal['name'] for goal in self.active_goals]}")

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

        This method reads global information (food availability, time of day, weather) from the World
        and also scans the immediate surroundings (e.g., 1-cell radius) on the grid for items like food
        and other agents. Sensory noise is now applied, meaning perceptions might be imperfect.
        Important perceptions are also added to the working memory buffer.
        An attention system can modify perception accuracy for specific stimuli.
        """
        if self.environment:
            # Global perceptions
            self.perception["food_available_global"] = self.environment.food_available
            self.perception["time_of_day"] = self.environment.time_of_day
            self.perception["current_weather"] = self.environment.current_weather  # New: Get current weather
            self.current_time_step = self.environment.time_step

            # Local grid perception (e.g., 1-cell radius around the agent)
            # This returns the raw grid view, not yet processed for other agents.
            raw_local_grid_view = self._get_local_grid_view(radius=1)

            # Apply sensory noise to local grid view with attention modulation
            self.perception["local_grid_view"] = []
            self.perception["food_in_sight"] = False
            for r_idx, row_content in enumerate(raw_local_grid_view):
                processed_row = []
                for c_idx, cell_content in enumerate(row_content):
                    # Adjust perception accuracy based on attention focus
                    current_perception_accuracy = self.perception_accuracy
                    if self.attention_focus == 'food' and cell_content == 'food':
                        current_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)  # Boost food perception
                    elif self.attention_focus == 'other_agents' and cell_content == 'agent':  # Assuming 'agent' placeholder for other agents
                        current_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)  # Boost agent perception

                    if random.random() < current_perception_accuracy:
                        processed_row.append(cell_content)  # Perceive correctly
                        if cell_content == 'food':
                            self.perception["food_in_sight"] = True
                            # Add perceived food location to working memory
                            # Calculate absolute grid coordinates for the food item
                            abs_r = self.pos_x + (r_idx - 1)  # r_idx 0,1,2 maps to offset -1,0,1
                            abs_c = self.pos_y + (c_idx - 1)
                            self.working_memory_buffer.append(
                                {"type": "perceived_food", "location": (abs_r, abs_c), "time": self.current_time_step})
                    else:
                        processed_row.append('unknown')  # Perceive incorrectly (noise/occlusion)
                self.perception["local_grid_view"].append(processed_row)

            # Check for other agents in local view (also apply noise with attention modulation)
            self.perception["other_agents_in_sight"] = []
            grid_size = len(self.environment.grid)  # Assuming square grid
            radius = 1  # Using the same radius as _get_local_grid_view

            for r_offset in range(-radius, radius + 1):
                for c_offset in range(-radius, radius + 1):  # Corrected loop range to radius + 1
                    view_row, view_col = self.pos_x + r_offset, self.pos_y + c_offset

                    # Skip self position
                    if view_row == self.pos_x and view_col == self.pos_y:
                        continue

                    if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                        # Adjust perception accuracy for other agents based on attention focus
                        current_agent_perception_accuracy = self.perception_accuracy
                        if self.attention_focus == 'other_agents':
                            current_agent_perception_accuracy = min(1.0,
                                                                    self.perception_accuracy + 0.2)  # Boost agent perception

                        # Apply noise to perception of other agents
                        if random.random() < current_agent_perception_accuracy:
                            # Iterate through all agents in the world to see if any are at this cell
                            for other_agent in self.environment.agents:
                                if other_agent is not self and other_agent.pos_x == view_row and other_agent.pos_y == view_col:
                                    agent_info = {"name": other_agent.name, "pos_x": other_agent.pos_x,
                                                  "pos_y": other_agent.pos_y}
                                    self.perception["other_agents_in_sight"].append(agent_info)
                                    # Add perceived other agent to working memory
                                    self.working_memory_buffer.append(
                                        {"type": "perceived_agent", "info": agent_info, "time": self.current_time_step})
                                    # Break inner loop once an agent is found at this cell to avoid duplicates
                                    break
                                    # Else: agent fails to perceive the other agent due to noise/occlusion
                    # Else: it's a boundary, no agent there

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
                             Note: This raw view does not include agents; agent perception handles that.
        """
        view = []
        grid_size = len(self.environment.grid)  # Assuming square grid

        for r_offset in range(-radius, radius + 1):
            row_view = []
            for c_offset in range(-radius, radius + 1):
                view_row, view_col = self.pos_x + r_offset, self.pos_y + c_offset
                if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                    # FIX: Corrected the index from 'cell_content' to 'view_col'
                    row_view.append(self.environment.grid[view_row][view_col])
                else:
                    row_view.append('boundary')  # Mark cells outside the grid
            view.append(row_view)
        return view

    def think(self) -> str:
        """
        Determines the agent's next action based on its internal state, emotions,
        learning algorithms, and now, the presence of other agents, memory recall,
        weather conditions, curiosity for exploration, active goals, working memory,
        and current attention focus.

        The thinking process involves updating physiological states, emotional states,
        and then using a combination of motivation, DQN, and memory to select an action.

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
        # This part is currently not used for the final action decision as DQN overrides it.
        _ = self.motivation_engine.decide_action(
            perception=self.perception,
            memory=self.short_term_memory,
            emotions=self.emotion_state
        )

        # --- Set Attention Focus ---
        # Prioritize attention based on most pressing needs or active goals
        if self.internal_state.hunger > 0.7:
            self.attention_focus = 'food'
            print(f"{self.name} is very hungry, focusing attention on food.")
        elif self.internal_state.fatigue > 0.7:
            self.attention_focus = 'rest'  # Could be 'shelter' or 'safe_spot' if those existed
            print(f"{self.name} is very fatigued, focusing attention on rest.")
        elif self.emotion_state.get("curiosity") > 0.6 and not self.active_goals:
            self.attention_focus = 'curiosity'
            print(f"{self.name} is very curious, focusing attention on exploration.")
        elif any(not goal["completed"] for goal in self.active_goals):
            # If there are active goals, focus on the highest priority uncompleted goal
            uncompleted_goals = [g for g in self.active_goals if not g["completed"]]
            if uncompleted_goals:
                highest_priority_goal = max(uncompleted_goals, key=lambda g: g["priority"])
                if highest_priority_goal["type"] == "reach_location":
                    self.attention_focus = 'location_target'  # Specific focus for location goals
                    print(f"{self.name} focusing attention on goal: {highest_priority_goal['name']}.")
                elif highest_priority_goal["type"] == "maintain_hunger_low":
                    self.attention_focus = 'food'  # Focus on food to maintain hunger
                    print(f"{self.name} focusing attention on goal: {highest_priority_goal['name']}.")
                else:
                    self.attention_focus = None  # Default if goal type not specifically handled
            else:
                self.attention_focus = None
        else:
            self.attention_focus = None  # Default/diffuse attention

        # 5. DQN Learner selects an action (primary decision maker for now).
        selected_action = self.q_learner.choose_action(
            hunger=self.internal_state.hunger,
            fatigue=self.internal_state.fatigue
        )

        # --- Memory Influence on Decision ---
        recalled_actions_for_hunger = []
        recalled_actions_for_fatigue = []

        for episode in self.episodic_memory.get_memory():
            is_recent_or_impactful = (self.current_time_step - episode['step'] <= 5) or \
                                     (episode['emotional_weight'] > 0.7)  # High emotional weight

            if is_recent_or_impactful:
                # If the agent was hungry in that past state and took 'seek_food'
                if episode['state']['hunger'] > 0.6 and episode['action'] == 'seek_food':
                    recalled_actions_for_hunger.append(episode)
                # If the agent was fatigued in that past state and took 'rest'
                if episode['state']['fatigue'] > 0.6 and episode['action'] == 'rest':
                    recalled_actions_for_fatigue.append(episode)

        # Simple decision modification based on recalled memories
        if self.internal_state.hunger > 0.6 and recalled_actions_for_hunger:
            # Check if seeking food was generally successful in recalled episodes
            successful_food_seeking_count = sum(1 for ep in recalled_actions_for_hunger if
                                                ep['state']['hunger'] > self.internal_state.hunger)  # Hunger reduced

            if successful_food_seeking_count > len(recalled_actions_for_hunger) / 2:  # More than half were successful
                if selected_action != "seek_food":
                    print(f"{self.name} recalls successful food seeking. Overriding to 'seek_food'.")
                    selected_action = "seek_food"  # Prioritize seeking food based on positive memory

        elif self.internal_state.fatigue > 0.6 and recalled_actions_for_fatigue:
            # Check if resting was generally successful in recalled episodes
            successful_rest_count = sum(1 for ep in recalled_actions_for_fatigue if
                                        ep['state']['fatigue'] > self.internal_state.fatigue)  # Fatigue reduced

            if successful_rest_count > len(recalled_actions_for_fatigue) / 2:  # More than half were successful
                if selected_action != "rest":
                    print(f"{self.name} recalls successful resting. Overriding to 'rest'.")
                    selected_action = "rest"  # Prioritize resting based on positive memory

        # --- Simple decision modification based on other agents (Example) ---
        if self.perception["other_agents_in_sight"] and self.internal_state.hunger > 0.5:
            # This is a very basic example. You could make it more complex:
            # - If other agent is also hungry, maybe compete or cooperate.
            # - If other agent is a 'threat', maybe move away.
            if selected_action == "explore":  # If DQN chose explore, but hungry and sees others
                print(f"{self.name} sees other agents and is hungry. Prioritizing seeking food over exploring.")
                selected_action = "seek_food"  # Override to seek food

        # --- Weather Influence on Decision ---
        current_weather = self.perception["current_weather"]
        if current_weather == "stormy":
            # In stormy weather, prioritize resting or seeking shelter (if shelter action existed)
            # and discourage exploration or long movements.
            if selected_action == "explore" or selected_action.startswith("move_"):
                print(f"{self.name} detects stormy weather. Prioritizing rest over movement/exploration.")
                selected_action = "rest"
            # Optionally, increase fatigue faster in stormy weather in internal_state.update()
            # or modify reward for certain actions.
        elif current_weather == "rainy":
            # In rainy weather, maybe slightly increase fatigue or reduce exploration reward
            if selected_action == "explore":
                if random.random() < 0.5:  # 50% chance to reconsider exploration
                    print(f"{self.name} detects rainy weather. Might reconsider exploration.")
                    selected_action = "rest" if self.internal_state.fatigue < 0.8 else "seek_food"  # Fallback to needs

        # --- Curiosity-Driven Exploration ---
        # Get the key for the current state (using DQNLearner's method for state representation)
        # Convert to bytes for defaultdict key
        current_state_key = self.q_learner.get_state_representation(self.internal_state.hunger,
                                                                    self.internal_state.fatigue).cpu().numpy().tobytes()

        # Increment visit count for the current state
        self.visited_states[current_state_key] += 1

        # Calculate an 'unfamiliarity' score for the current state (e.g., inverse of visit count)
        # Add a small epsilon to avoid division by zero for unvisited states
        unfamiliarity_score = 1.0 / (self.visited_states[current_state_key] + 1.0)

        # Combine curiosity emotion with unfamiliarity score
        # A higher curiosity emotion and higher unfamiliarity score will increase the chance of exploration
        curiosity_threshold = 0.5  # Base threshold for curiosity to influence action
        curiosity_influence = self.emotion_state.get("curiosity") * unfamiliarity_score

        # If curiosity is high enough and basic needs are not critical, encourage exploration
        if curiosity_influence > curiosity_threshold and \
                self.internal_state.hunger < 0.7 and self.internal_state.fatigue < 0.7:

            # If the DQN didn't choose explore, but curiosity is high,
            # consider overriding to an exploration action (e.g., a random move)
            if selected_action not in ["explore", "move_up", "move_down", "move_left", "move_right"]:
                print(f"{self.name} is curious! Overriding to exploration due to unfamiliarity.")
                # Choose a random movement action for exploration
                selected_action = random.choice(["move_up", "move_down", "move_left", "move_right"])
            elif selected_action == "explore":
                print(f"{self.name} is already exploring and curious.")

        # --- Goal System Influence ---
        # Iterate through active goals and potentially modify the selected action
        for goal in self.active_goals:
            if goal["completed"]:
                continue  # Skip completed goals

            if goal["type"] == "reach_location":
                # Calculate distance to target
                dist_x = abs(self.pos_x - goal["target_x"])
                dist_y = abs(self.pos_y - goal["target_y"])
                distance = dist_x + dist_y  # Manhattan distance

                if distance == 0:
                    goal["completed"] = True
                    print(f"{self.name} completed goal: {goal['name']} at ({self.pos_x},{self.pos_y})!")
                    # You could add a reward here for goal completion
                    self.last_action_reward += 1.0  # Bonus reward for goal completion
                    continue

                # If agent is close to goal and not critically hungry/fatigued, prioritize movement towards it
                if distance <= 3 and self.internal_state.hunger < 0.8 and self.internal_state.fatigue < 0.8:
                    print(f"{self.name} is prioritizing goal: {goal['name']}. Distance: {distance}")
                    # Decide which movement direction reduces distance most
                    if dist_x > 0:
                        if self.pos_x < goal["target_x"]:
                            selected_action = "move_down"
                        else:
                            selected_action = "move_up"
                    elif dist_y > 0:
                        if self.pos_y < goal["target_y"]:
                            selected_action = "move_right"
                        else:
                            selected_action = "move_left"
                    # If already at target x or y, prioritize the other dimension
                    # This simple logic might need refinement for optimal pathfinding, but provides goal-driven movement.
                    break  # Prioritize one goal at a time if multiple are active and relevant

            elif goal["type"] == "maintain_hunger_low":
                if self.internal_state.hunger < goal["threshold"]:
                    goal["current_duration"] += 1
                    if goal["current_duration"] >= goal["duration_steps"]:
                        goal["completed"] = True
                        print(
                            f"{self.name} completed goal: {goal['name']} (maintained hunger below {goal['threshold']:.2f} for {goal['duration_steps']} steps)!")
                        self.last_action_reward += 0.8  # Reward for maintaining state
                    # If close to threshold, prioritize seek_food
                    if self.internal_state.hunger >= goal["threshold"] * 0.8 and selected_action != "seek_food":
                        print(f"{self.name} prioritizing goal: {goal['name']}. Hunger getting high, seeking food.")
                        selected_action = "seek_food"
                else:
                    goal["current_duration"] = 0  # Reset if hunger goes above threshold

        # --- Working Memory Influence ---
        # Example: If working memory contains a recent food location and agent is hungry,
        # prioritize moving towards that food.
        if self.internal_state.hunger > 0.6:  # If agent is hungry
            for item in reversed(self.working_memory_buffer):  # Check most recent items first
                if item["type"] == "perceived_food" and \
                        self.current_time_step - item["time"] <= 3:  # If food was seen recently (e.g., within 3 steps)
                    food_x, food_y = item["location"]
                    # If food is not at current location and not already moving towards it
                    if (self.pos_x, self.pos_y) != (food_x, food_y) and \
                            not selected_action.startswith("move_"):  # Avoid overriding an existing move

                        print(
                            f"{self.name} recalls food at {food_x},{food_y} from working memory. Prioritizing movement.")
                        # Simple movement logic towards recalled food
                        if abs(self.pos_x - food_x) > abs(self.pos_y - food_y):
                            if self.pos_x < food_x:
                                selected_action = "move_down"
                            else:
                                selected_action = "move_up"
                        else:
                            if self.pos_y < food_y:
                                selected_action = "move_right"
                            else:
                                selected_action = "move_left"
                        break  # Prioritize this recalled food location

        self._last_performed_action = selected_action
        return selected_action

    def act(self, action: str):
        """
        Executes the chosen action, updates the agent's internal state and DQN,
        and records the experience in episodic memory.
        This method now includes movement logic for grid-based actions.

        Args:
            action (str): The action decided by the agent's motivation engine.
                          Expected actions: "seek_food", "rest", "explore",
                          "move_up", "move_down", "move_left", "move_right".
        """
        # Take a snapshot of the agent's internal state before the action.
        previous_internal_state = self.internal_state.snapshot()
        original_pos_x, original_pos_y = self.pos_x, self.pos_y  # Store original position

        # Perform the selected action and apply its effects.
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

        # --- Movement Actions ---
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
                # You could add obstacle checks here later.
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
        # Pass the mood_value to the reward learner to adjust the reward
        adjusted_reward = self.reward_learner.update(
            previous_internal_state,
            self.internal_state,
            action,
            self.last_action_reward,  # Pass the raw reward
            self.internal_state.mood_value  # Pass the current mood value
        )
        # Update the agent's last_action_reward with the adjusted value
        self.last_action_reward = adjusted_reward

        # Update DQN with the experience using the adjusted reward
        self.q_learner.update(
            previous_internal_state.hunger, previous_internal_state.fatigue,
            action,
            self.last_action_reward,  # Use the adjusted reward here
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
        DQN learner info, and episodic memory sample.
        """
        print(f"{self.name} at ({self.pos_x},{self.pos_y}) → {self.internal_state}")
        print("Emotions:", self.emotion_state)
        print("DQN Learner Info:")  # Changed from Q-table to DQN Learner
        print(self.q_learner)
        print("Episodic Memory (sample):")
        print(self.episodic_memory)
        print("Local Perception (3x3 view):")
        # Print local grid view for debugging
        if self.perception["local_grid_view"]:
            for row_content in self.perception["local_grid_view"]:
                print(" ".join(row_content))
        else:
            print("No local view available (sense() not called or environment not set).")

        if self.perception["other_agents_in_sight"]:
            print("Other agents in sight:")
            for agent_info in self.perception["other_agents_in_sight"]:
                print(f"  - {agent_info['name']} at ({agent_info['pos_x']},{agent_info['pos_y']})")
        else:
            print("No other agents in sight.")

        print(f"Current Weather: {self.perception['current_weather']}")  # Log current weather

        # Log active goals
        if self.active_goals:
            print("Active Goals:")
            # Iterate through a copy of active_goals to allow modification during iteration if needed
            for goal in list(self.active_goals):  # Use list() to iterate over a copy
                status = "Completed" if goal["completed"] else "Active"
                if goal["type"] == "reach_location":
                    print(
                        f"  - {goal['name']}: {status}, Target: ({goal['target_x']},{goal['target_y']}), Priority: {goal['priority']:.2f}")
                elif goal["type"] == "maintain_hunger_low":
                    print(
                        f"  - {goal['name']}: {status}, Threshold: {goal['threshold']:.2f}, Duration: {goal['current_duration']}/{goal['duration_steps']}, Priority: {goal['priority']:.2f}")
        else:
            print("No active goals.")

        # Log working memory content
        if self.working_memory_buffer:
            print("Working Memory:")
            for item in self.working_memory_buffer:
                print(f"  - Type: {item['type']}, Time: {item['time']}")
                if item['type'] == 'perceived_food':
                    print(f"    Location: {item['location']}")
                elif item['type'] == 'perceived_agent':
                    print(f"    Agent: {item['info']['name']} at ({item['info']['pos_x']},{item['info']['pos_y']})")
        else:
            print("Working Memory is empty.")

        print(f"Attention Focus: {self.attention_focus}")  # Log current attention focus

