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
from typing import List, Tuple, Dict, Any

# Import Consciousness States
from core.consciousness.base import ConsciousnessState
from core.consciousness.awake_state import AwakeState
from core.consciousness.asleep_state import AsleepState
from core.consciousness.focused_state import FocusedState

# Import Memory Modules
from core.memory.semantic_memory import SemanticMemory
from core.memory.procedural_memory import ProceduralMemory

# Import DecisionMaker
from core.decision.decision_maker import DecisionMaker

# New: Import Cognitive Modules
from core.cognitive_modules.base_module import CognitiveModule
from core.cognitive_modules.problem_solver import ProblemSolver  # Example module
from core.cognitive_modules.goal_generator import GoalGenerator  # New: Import GoalGenerator


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
    a basic attention system, uses a Deep Q-Network (DQN) for learning,
    generates an internal monologue or reasoning trace, manages different
    states of consciousness, includes semantic and procedural memory,
    incorporates a hierarchical decision-maker with deliberative/reactive modes,
    supports plug-and-play cognitive modules, has the ability to
    interact with and move objects in the environment, supports
    more complex goal hierarchies, and can now dynamically generate and
    modify goals.
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
            "current_weather": "sunny",  # New: Current weather condition
            "local_grid_view": [],  # Agent's view of its immediate surroundings
            "food_in_sight": False,  # Whether food is in agent's local view
            "other_agents_in_sight": [],  # List of (agent_name, pos_x, pos_y) of other agents in local view
            "obstacle_in_sight": False,  # New: Whether an obstacle is in agent's local view
            "obstacle_locations": []  # New: List of (row, col) of obstacles in local view
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
        # Semantic memory for general knowledge
        self.semantic_memory: SemanticMemory = SemanticMemory()
        # Procedural memory for learned skills and habits
        self.procedural_memory: ProceduralMemory = ProceduralMemory()

        # High-level decision maker
        self.decision_maker: DecisionMaker = DecisionMaker()

        # Learner for generic reward-based learning (e.g., policy gradients - though not fully implemented here).
        self.reward_learner: RewardLearner = RewardLearner()
        # Snapshot of the agent's state before an action, used for learning.
        self._previous_state_snapshot = None

        # Changed: Use DQNLearner instead of QTableLearner
        # State size for DQN is 2 (hunger, fatigue)
        self.q_learner: DQNLearner = DQNLearner(
            actions=["seek_food", "rest", "explore", "move_up", "move_down", "move_left", "move_right", "move_object"],
            # New: Added "move_object" action
            state_size=2  # Hunger and Fatigue
        )

        # Physiological state values (hunger, fatigue) before the last action, for reward calculation.
        self.previous_physiological_states = None
        # The last action performed by the agent.
        self._last_performed_action = None
        # The reward received from the last action.
        self.last_action_reward: float = 0.0

        # Manages the agent's emotional state.
        self.emotion_state: EmotionState = EmotionState()
        # Strategy for updating the agent's emotions.
        self.emotion_strategy: BasicEmotionStrategy = BasicEmotionStrategy(self.emotion_state)

        # New: Perception accuracy for sensory noise
        self.perception_accuracy: float = perception_accuracy

        # New: Track visited states for curiosity
        self.visited_states: Dict[str, int] = defaultdict(int)  # Counts how many times a state has been visited

        # New: Basic Goal System (now with hierarchical capabilities)
        # Goals can be represented as dictionaries:
        # {
        #   "id": "unique_goal_id",
        #   "type": "reach_location" | "maintain_hunger_low" | "clear_path" | "explore_area",
        #   "name": "Descriptive Name",
        #   "priority": 0.0-1.0,
        #   "completed": False,
        #   "parent_goal_id": None | "id_of_parent_goal", # New: For hierarchical goals
        #   "prerequisites": [], # New: List of goal IDs that must be completed first
        #   "sub_goals": [], # New: List of goal IDs that are part of this goal
        #   "target_x": int, "target_y": int, # For reach_location
        #   "threshold": float, "duration_steps": int, "current_duration": int # For maintain_hunger_low
        #   "obstacle_location": (int, int) # For clear_path
        # }
        self.active_goals: List[Dict] = []
        self._initialize_goals()  # Add some initial goals

        # New: Working Memory Buffer
        # Stores recent important perceptions or thoughts for short-term recall.
        # Max length of 5 items, oldest items are discarded when new ones are added.
        self.working_memory_buffer: deque = deque(maxlen=5)

        # New: Attention System
        # What the agent is currently focusing its attention on.
        # Can be 'food', 'other_agents', 'curiosity', 'goal', 'obstacle', or None (default/diffuse attention).
        self.attention_focus: str = None

        # New: Internal Monologue / Reasoning Trace
        self.internal_monologue: str = ""

        # New: Consciousness States
        self.awake_state = AwakeState(self)
        self.asleep_state = AsleepState(self)
        self.focused_state = FocusedState(self)
        self.current_consciousness_state: ConsciousnessState = self.awake_state  # Start in Awake state
        self.current_consciousness_state.enter()  # Call enter for initial state

        # New: Plug-and-Play Cognitive Modules
        # Store cognitive modules in a dictionary for easy access by name
        self.cognitive_modules: Dict[str, CognitiveModule] = {}
        self._initialize_cognitive_modules()

    def _initialize_goals(self):
        """
        Initializes a set of default goals for the agent, now supporting hierarchies.
        """
        # Goal 1: Clear a path to a specific location (Main Goal)
        # This goal will have sub-goals for clearing obstacles and then reaching the location.
        main_goal_reach_center = {
            "id": "goal_reach_center",
            "type": "reach_location",
            "name": "Reach Center of Map",
            "priority": 0.9,  # High priority main goal
            "completed": False,
            "parent_goal_id": None,
            "prerequisites": [],  # No prerequisites for the main goal itself
            "target_x": 5,
            "target_y": 5
        }
        self.active_goals.append(main_goal_reach_center)

        # Sub-goal 1.1: Clear Obstacle (Prerequisite for reaching center if an obstacle is in the way)
        # This goal will be dynamically created/prioritized by DecisionMaker/ProblemSolver
        # if an obstacle is detected on the path to "goal_reach_center".
        # For now, we'll add a placeholder that DecisionMaker can activate.
        # This goal won't be active initially, but DecisionMaker will check for it.
        clear_obstacle_sub_goal = {
            "id": "sub_goal_clear_obstacle",
            "type": "clear_path",  # New type for clearing obstacles
            "name": "Clear Obstacle on Path",
            "priority": 0.85,  # Slightly lower than main goal, but still high
            "completed": False,
            "parent_goal_id": "goal_reach_center",
            "prerequisites": [],  # No specific prerequisites other than an obstacle existing
            "obstacle_location": None  # This will be set dynamically when an obstacle is identified
        }
        self.active_goals.append(clear_obstacle_sub_goal)

        # Goal 2: Maintain low hunger for a period (Independent Goal)
        self.active_goals.append({
            "id": "goal_stay_fed",
            "type": "maintain_hunger_low",
            "name": "Stay Fed",
            "priority": 0.6,
            "completed": False,
            "parent_goal_id": None,
            "prerequisites": [],
            "threshold": 0.3,
            "duration_steps": 20,
            "current_duration": 0
        })
        print(f"{self.name} initialized with goals: {[goal['name'] for goal in self.active_goals]}")

    def _initialize_cognitive_modules(self):
        """
        Initializes and registers plug-and-play cognitive modules.
        """
        # Add the ProblemSolver module
        problem_solver = ProblemSolver(self)
        self.cognitive_modules[problem_solver.get_module_name()] = problem_solver
        print(f"{self.name} initialized with cognitive module: {problem_solver.get_module_name()}")

        # New: Add the GoalGenerator module
        goal_generator = GoalGenerator(self)
        self.cognitive_modules[goal_generator.get_module_name()] = goal_generator
        print(f"{self.name} initialized with cognitive module: {goal_generator.get_module_name()}")
        # Add other modules here as they are created:
        # social_module = SocialModule(self)
        # self.cognitive_modules[social_module.get_module_name()] = social_module

    def set_environment(self, env):
        """
        Sets the simulation environment for the agent.

        Args:
            env: An instance of the simulation environment (e.g., World class).
        """
        self.environment = env

    def transition_to_state(self, new_state: ConsciousnessState):
        """
        Transitions the agent to a new state of consciousness.

        Args:
            new_state (ConsciousnessState): The new consciousness state to transition to.
        """
        if self.current_consciousness_state != new_state:
            self.current_consciousness_state.exit()
            self.current_consciousness_state = new_state
            self.current_consciousness_state.enter()
            print(f"{self.name} transitioned to {new_state.get_state_name()} state.")
            self.internal_monologue += f"Transitioning to {new_state.get_state_name()} state. "

    # --- Default (Awake) Behavior Methods ---
    # These methods encapsulate the original sense/think/act logic,
    # which will be called by the AwakeState.
    def _sense_default(self):
        """
        Default sensing behavior for the agent (used by AwakeState).
        """
        if self.environment:
            # Global perceptions
            self.perception["food_available_global"] = self.environment.food_available
            self.perception["time_of_day"] = self.environment.time_of_day
            self.perception["current_weather"] = self.environment.current_weather
            self.current_time_step = self.environment.time_step

            # Local grid perception (e.g., 1-cell radius around the agent)
            raw_local_grid_view = self._get_local_grid_view(radius=1)

            # Apply sensory noise to local grid view with attention modulation
            self.perception["local_grid_view"] = []
            self.perception["food_in_sight"] = False
            self.perception["obstacle_in_sight"] = False  # New: Reset obstacle flag
            self.perception["obstacle_locations"] = []  # New: Reset obstacle locations

            for r_idx, row_content in enumerate(raw_local_grid_view):
                processed_row = []
                for c_idx, cell_content in enumerate(row_content):
                    # Adjust perception accuracy based on attention focus
                    current_perception_accuracy = self.perception_accuracy
                    if self.attention_focus == 'food' and cell_content == 'food':
                        current_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)  # Boost food perception
                    elif self.attention_focus == 'other_agents' and cell_content == 'agent':
                        current_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)  # Boost agent perception
                    elif self.attention_focus == 'obstacle' and cell_content == 'obstacle':  # New: Boost obstacle perception
                        current_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)
                        # You can add more rules for other attention focuses

                    if random.random() < current_perception_accuracy:
                        processed_row.append(cell_content)  # Perceive correctly
                        if cell_content == 'food':
                            self.perception["food_in_sight"] = True
                            # Add perceived food location to working memory
                            abs_r = self.pos_x + (r_idx - 1)
                            abs_c = self.pos_y + (c_idx - 1)
                            self.working_memory_buffer.append(
                                {"type": "perceived_food", "location": (abs_r, abs_c), "time": self.current_time_step})
                        elif cell_content == 'obstacle':  # New: Perceive obstacle
                            self.perception["obstacle_in_sight"] = True
                            abs_r = self.pos_x + (r_idx - 1)
                            abs_c = self.pos_y + (c_idx - 1)
                            self.perception["obstacle_locations"].append((abs_r, abs_c))
                            self.working_memory_buffer.append({"type": "perceived_obstacle", "location": (abs_r, abs_c),
                                                               "time": self.current_time_step})
                    else:
                        processed_row.append('unknown')  # Perceive incorrectly (noise/occlusion)
                self.perception["local_grid_view"].append(processed_row)

            # Check for other agents in local view (also apply noise with attention modulation)
            self.perception["other_agents_in_sight"] = []
            grid_size = len(self.environment.grid)
            radius = 1

            for r_offset in range(-radius, radius + 1):
                for c_offset in range(-radius, radius + 1):
                    view_row, view_col = self.pos_x + r_offset, self.pos_y + c_offset

                    # Skip self position
                    if view_row == self.pos_x and view_col == self.pos_y:
                        continue

                    if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                        # Adjust perception accuracy for other agents based on attention focus
                        current_agent_perception_accuracy = self.perception_accuracy
                        if self.attention_focus == 'other_agents':
                            current_agent_perception_accuracy = min(1.0, self.perception_accuracy + 0.2)

                        # Apply noise to perception of other agents
                        if random.random() < current_agent_perception_accuracy:
                            for other_agent in self.environment.agents:
                                if other_agent is not self and other_agent.pos_x == view_row and other_agent.pos_y == view_col:
                                    agent_info = {"name": other_agent.name, "pos_x": other_agent.pos_x,
                                                  "pos_y": other_agent.pos_y}
                                    self.perception["other_agents_in_sight"].append(agent_info)
                                    # Add perceived other agent to working memory
                                    self.working_memory_buffer.append(
                                        {"type": "perceived_agent", "info": agent_info, "time": self.current_time_step})
                                    break
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
                    row_view.append(self.environment.grid[view_row][view_col])
                else:
                    row_view.append('boundary')  # Mark cells outside the grid
            view.append(row_view)
        return view

    def _think_default(self, decision_mode: str) -> str:  # Added decision_mode parameter
        """
        Default thinking behavior for the agent (used by AwakeState and FocusedState).
        This method now primarily orchestrates the decision-making process by
        delegating to the DecisionMaker, based on the current decision mode.

        Args:
            decision_mode (str): The current decision-making mode ('reactive' or 'deliberative').

        Returns:
            str: The chosen action.
        """
        # 1. Update physiological state (e.g., hunger, fatigue increases over time).
        time_delta_factor = 1.5 if self.perception["time_of_day"] == "night" else 1.0
        self.internal_state.update(delta_time=time_delta_factor)

        # 2. Update emotional state based on environmental perceptions and internal state.
        self.emotion_strategy.update_emotions(self.perception, self.internal_state)

        # 3. Store current state before acting (for learning)
        self.previous_physiological_states = (self.internal_state.hunger, self.internal_state.fatigue)

        # 4. Emotion-driven motivation engine decides preferred action (not directly used for final action here)
        _ = self.motivation_engine.decide_action(
            perception=self.perception,
            memory=self.short_term_memory,
            emotions=self.emotion_state
        )

        # --- Set Attention Focus ---
        # Prioritize attention based on most pressing needs or active goals
        if self.internal_state.hunger > 0.7:
            self.attention_focus = 'food'
            self.internal_monologue += "I am very hungry, so I should focus on finding food. "
        elif self.internal_state.fatigue > 0.7:
            self.attention_focus = 'rest'  # Could be 'shelter' or 'safe_spot' if those existed
            self.internal_monologue += "I am very fatigued, so I should focus on resting. "
        elif self.emotion_state.get("curiosity") > 0.6 and not self.active_goals:
            self.attention_focus = 'curiosity'
            self.internal_monologue += "I feel curious and have no immediate goals, so I will focus on exploration. "
        elif any(not goal["completed"] for goal in self.active_goals):
            uncompleted_goals = [g for g in self.active_goals if not g["completed"]]
            if uncompleted_goals:
                highest_priority_goal = max(uncompleted_goals, key=lambda g: g["priority"])
                if highest_priority_goal["type"] == "reach_location":
                    self.attention_focus = 'location_target'  # Specific focus for location goals
                    self.internal_monologue += f"My top priority is to reach {highest_priority_goal['name']} at ({highest_priority_goal['target_x']},{highest_priority_goal['target_y']}). "
                elif highest_priority_goal["type"] == "maintain_hunger_low":
                    self.attention_focus = 'food'  # Focus on food to maintain hunger
                    self.internal_monologue += f"I need to maintain low hunger for {highest_priority_goal['name']}, so I will focus on food. "
                elif highest_priority_goal["type"] == "clear_path":  # New: Focus for clear_path goal
                    self.attention_focus = 'obstacle'
                    self.internal_monologue += f"My top priority is to clear a path for {highest_priority_goal['name']}. "
                else:
                    self.attention_focus = None  # Default if goal type not specifically handled
                    self.internal_monologue += "I have an active goal but no specific focus for it. "
            else:
                self.attention_focus = None
                self.internal_monologue += "No active goals to focus on. "
        elif self.perception["obstacle_in_sight"]:  # New: Focus on obstacle if present
            self.attention_focus = 'obstacle'
            self.internal_monologue += "I see an obstacle, focusing on how to deal with it. "
        else:
            self.attention_focus = None  # Default/diffuse attention
            self.internal_monologue += "No specific attention focus right now. "

        # --- Process Cognitive Modules ---
        # Cognitive modules can provide inputs or suggestions to the DecisionMaker.
        # Collect outputs from all active cognitive modules.
        cognitive_module_outputs: Dict[str, Any] = {}
        for module_name, module_instance in self.cognitive_modules.items():
            module_output = module_instance.process()
            if module_output:
                cognitive_module_outputs[module_name] = module_output
                self.internal_monologue += f"Cognitive module '{module_name}' provided output: {module_output}. "

        # New: Process GoalGenerator output
        if "GoalGenerator" in self.cognitive_modules:
            goal_generator_output = cognitive_module_outputs.get("GoalGenerator", {})
            if "new_goals" in goal_generator_output:
                for new_goal in goal_generator_output["new_goals"]:
                    # Check if a goal with the same ID already exists to prevent duplicates
                    if not any(g["id"] == new_goal["id"] for g in self.active_goals):
                        self.active_goals.append(new_goal)
                        self.internal_monologue += f"GoalGenerator: Added new goal '{new_goal['name']}'. "
            if "modify_goals" in goal_generator_output:
                for modify_instruction in goal_generator_output["modify_goals"]:
                    goal_id_to_modify = modify_instruction["id"]
                    for goal in self.active_goals:
                        if goal["id"] == goal_id_to_modify:
                            for key, value in modify_instruction.items():
                                if key != "id":  # Don't modify the ID
                                    goal[key] = value
                            self.internal_monologue += f"GoalGenerator: Modified goal '{goal['name']}'. "
                            break

        # --- Decision Maker determines the final action based on the mode ---
        # All complex decision logic (DQN, memories, goals, environment, emotions)
        # is now encapsulated within the DecisionMaker.
        final_action = self.decision_maker.decide_final_action(self, decision_mode)

        self.internal_monologue += f"Therefore, I have decided to {final_action}. "
        self._last_performed_action = final_action
        return final_action

    def sense(self):
        """
        Delegates sensing to the current consciousness state.
        """
        self.current_consciousness_state.sense()

    def think(self) -> str:
        """
        Delegates thinking to the current consciousness state.
        Also handles transitions between consciousness states based on internal conditions.

        Returns:
            str: The chosen action.
        """
        # Determine if a state transition is needed BEFORE thinking
        if self.internal_state.fatigue > 0.9 and self.current_consciousness_state != self.asleep_state:
            self.transition_to_state(self.asleep_state)
        elif self.internal_state.fatigue < 0.2 and self.current_consciousness_state == self.asleep_state:
            self.transition_to_state(self.awake_state)
        elif self.internal_state.hunger > 0.8 and self.current_consciousness_state != self.focused_state:
            # If very hungry, try to enter focused state on food
            self.attention_focus = 'food'  # Ensure attention is on food
            self.transition_to_state(self.focused_state)
        elif self.internal_state.hunger < 0.3 and self.current_consciousness_state == self.focused_state and self.attention_focus == 'food':
            # If hunger is low and was focused on food, return to awake
            self.transition_to_state(self.awake_state)
        elif self.current_consciousness_state == self.focused_state and self.attention_focus == 'location_target':
            # If focused on a location goal and it's completed, return to awake
            target_goal = next((g for g in self.active_goals if g["type"] == "reach_location" and not g["completed"]),
                               None)
            if not target_goal:  # Goal completed or no active location goal
                self.transition_to_state(self.awake_state)
        # Add more complex transition logic here

        # Initialize internal monologue for this step
        self.internal_monologue = f"Time Step {self.current_time_step}: I am at ({self.pos_x},{self.pos_y}). "
        self.internal_monologue += f"Hunger: {self.internal_state.hunger:.2f}, Fatigue: {self.internal_state.fatigue:.2f}, Mood: {self.internal_state.mood}. "
        self.internal_monologue += f"My emotions are: {self.emotion_state}. "
        self.internal_monologue += f"Current consciousness state: {self.current_consciousness_state.get_state_name()}. "

        # Delegate thinking to the current consciousness state, passing the decision mode
        return self.current_consciousness_state.think()  # The consciousness state's think() will call _think_default with mode

    def act(self, action: str):
        """
        Delegates acting to the current consciousness state.

        Args:
            action (str): The action to perform.
        """
        self.current_consciousness_state.act(action)

    def _act_default(self, action: str):
        """
        Default action execution logic for the agent (used by various ConsciousnessStates).
        This method contains the core effects of performing an action on the agent's
        internal state, environment, and learning.

        Args:
            action (str): The action to perform.
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

            # Check if new position is within grid boundaries AND not an obstacle
            grid_size = len(self.environment.grid)
            if 0 <= new_pos_x < grid_size and 0 <= new_pos_y < grid_size:
                if self.environment.grid[new_pos_x][new_pos_y] != 'obstacle':  # Cannot move onto an obstacle
                    self.pos_x, self.pos_y = new_pos_x, new_pos_y
                    self.last_action_reward = -0.01  # Small cost for movement
                    print(f"{self.name} moved {action.replace('move_', '')} to ({self.pos_x},{self.pos_y}).")
                else:
                    self.last_action_reward = -0.1  # Penalty for trying to move onto an obstacle
                    print(
                        f"{self.name} tried to move {action.replace('move_', '')} onto an obstacle at ({new_pos_x},{new_pos_y}).")
            else:
                self.last_action_reward = -0.1  # Penalty for trying to move out of bounds
                print(
                    f"{self.name} tried to move {action.replace('move_', '')} out of bounds from ({original_pos_x},{original_pos_y}).")

        # --- New: Move Object Action ---
        elif action == "move_object":
            # Agent tries to move an object from its current position to an adjacent empty cell.
            # For simplicity, let's assume it tries to move an object one step to the right if possible.
            # This logic can be expanded to allow agent to choose direction.
            target_obj_x, target_obj_y = self.pos_x, self.pos_y  # Object is at agent's current position

            # Find an adjacent empty cell to move the object to
            # Prioritize right, then down, then left, then up
            possible_new_positions = [
                (target_obj_x, target_obj_y + 1),  # Right
                (target_obj_x + 1, target_obj_y),  # Down
                (target_obj_x, target_obj_y - 1),  # Left
                (target_obj_x - 1, target_obj_y)  # Up
            ]

            moved_successfully = False
            for new_obj_x, new_obj_y in possible_new_positions:
                if self.environment.move_object_at_position(target_obj_x, target_obj_y, new_obj_x, new_obj_y):
                    self.last_action_reward = 0.3  # Positive reward for moving an object
                    print(
                        f"{self.name} successfully moved object from ({target_obj_x},{target_obj_y}) to ({new_obj_x},{new_obj_y}).")
                    moved_successfully = True
                    break

            if not moved_successfully:
                self.last_action_reward = -0.2  # Penalty for failing to move object
                print(
                    f"{self.name} tried to move an object at ({target_obj_x},{target_obj_y}) but failed (no object or no empty adjacent cell).")

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
        print("DQN Learner Info:")
        print(self.q_learner)
        print("Episodic Memory (sample):")
        print(self.episodic_memory)
        print("Semantic Memory:")  # Log Semantic Memory
        print(self.semantic_memory)  # Print the semantic memory content
        print("Procedural Memory:")  # Log Procedural Memory
        print(self.procedural_memory)  # Print the procedural memory content
        print("Local Perception (3x3 view):")
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

        if self.perception["obstacle_in_sight"]:  # New: Log obstacle info
            print(f"Obstacles in sight at: {self.perception['obstacle_locations']}")
        else:
            print("No obstacles in sight.")

        print(f"Current Weather: {self.perception['current_weather']}")

        if self.active_goals:
            print("Active Goals:")
            for goal in list(self.active_goals):
                status = "Completed" if goal["completed"] else "Active"
                goal_details = f"  - {goal['name']}: {status}, Priority: {goal['priority']:.2f}"
                if goal["type"] == "reach_location":
                    goal_details += f", Target: ({goal['target_x']},{goal['target_y']})"
                elif goal["type"] == "maintain_hunger_low":
                    goal_details += f", Threshold: {goal['threshold']:.2f}, Duration: {goal['current_duration']}/{goal['duration_steps']}"
                elif goal["type"] == "clear_path":  # New: Display obstacle location for clear_path goal
                    if goal["obstacle_location"]:
                        goal_details += f", Obstacle: ({goal['obstacle_location'][0]},{goal['obstacle_location'][1]})"

                # Add parent and prerequisite info if available
                if goal.get("parent_goal_id"):
                    goal_details += f", Parent: {goal['parent_goal_id']}"
                if goal.get("prerequisites"):
                    goal_details += f", Prerequisites: {', '.join(goal['prerequisites'])}"

                print(goal_details)
        else:
            print("No active goals.")

        if self.working_memory_buffer:
            print("Working Memory:")
            for item in self.working_memory_buffer:
                print(f"  - Type: {item['type']}, Time: {item['time']}")
                if item['type'] == 'perceived_food':
                    print(f"    Location: {item['location']}")
                elif item['type'] == 'perceived_agent':
                    print(f"    Agent: {item['info']['name']} at ({item['info']['pos_x']},{item['info']['pos_y']})")
                elif item['type'] == 'perceived_obstacle':  # New: Log perceived obstacle in working memory
                    print(f"    Location: {item['location']}")
        else:
            print("Working Memory is empty.")

        print(f"Attention Focus: {self.attention_focus}")
        print(f"Internal Monologue: {self.internal_monologue}")

