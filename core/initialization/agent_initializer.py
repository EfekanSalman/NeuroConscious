# !/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING, List, Dict, Any
from collections import defaultdict, deque
from core.state import InternalState
from core.motivation.basic_motivation import BasicMotivationEngine
from core.mood.base import MoodStrategy
from core.memory.episodic import EpisodicMemory
from core.memory.semantic_memory import SemanticMemory
from core.memory.procedural_memory import ProceduralMemory
from core.learning.reward_learner import RewardLearner
from core.learning.dqn_learner import DQNLearner
from core.emotion.emotion_state import EmotionState
from core.emotion.basic_emotion import BasicEmotionStrategy
from core.decision.decision_maker import DecisionMaker
from core.consciousness.awake_state import AwakeState
from core.consciousness.asleep_state import AsleepState
from core.consciousness.focused_state import FocusedState
from core.cognitive_modules.problem_solver import ProblemSolver
from core.cognitive_modules.goal_generator import GoalGenerator
from core.perception.perception_manager import PerceptionManager
from core.action.action_executor import ActionExecutor
from core.thought.thought_processor import ThoughtProcessor

if TYPE_CHECKING:
    from agent.base_agent import Agent


class AgentInitializer:
    """
    Manages the initialization of various core components and attributes for an agent.

    This class centralizes the setup logic for an agent's internal state, memory systems,
    learning modules, cognitive modules, and consciousness states, reducing the
    complexity of the main Agent constructor.
    """

    def __init__(self, agent: 'Agent'):
        """
        Initializes the AgentInitializer with a reference to the agent.

        Args:
            agent (Agent): The agent instance to be initialized.
        """
        self.agent = agent

    def initialize_core_attributes(self, name: str, mood_strategy: MoodStrategy, perception_accuracy: float):
        """
        Initializes the fundamental attributes of the agent.

        Args:
            name (str): The unique name of the agent.
            mood_strategy (MoodStrategy): The strategy used to calculate the agent's mood.
            perception_accuracy (float): The probability (0.0 to 1.0) of correct perception.
        """
        self.agent.name = name
        self.agent.internal_state = InternalState(mood_strategy)
        self.agent.motivation_engine = BasicMotivationEngine(agent=self.agent)
        self.agent.perception = {
            "food_available_global": False,
            "water_available_global": False,
            "time_of_day": "day",
            "current_weather": "sunny",
            "local_grid_view": [],
            "food_in_sight": False,
            "water_in_sight": False,
            "water_locations": [],
            "other_agents_in_sight": [],
            "obstacle_in_sight": False,
            "obstacle_locations": []
        }
        self.agent.short_term_memory = {
            "food_last_seen": None,
            "water_last_seen": None
        }
        self.agent.environment = None
        self.agent.current_time_step = 0
        self.agent.pos_x = 0
        self.agent.pos_y = 0

        self.agent.episodic_memory = EpisodicMemory(capacity=5)
        self.agent.semantic_memory = SemanticMemory()
        self.agent.procedural_memory = ProceduralMemory()
        self.agent.decision_maker = DecisionMaker()

        self.agent.reward_learner = RewardLearner()
        self.agent._previous_state_snapshot = None
        self.agent.q_learner = DQNLearner(
            actions=["seek_food", "rest", "explore", "move_up", "move_down", "move_left", "move_right", "move_object",
                     "drink_water"],
            state_size=3
        )
        self.agent.previous_physiological_states = None
        self.agent._last_performed_action = None
        self.agent.last_action_reward = 0.0

        self.agent.emotion_state = EmotionState()
        self.agent.emotion_strategy = BasicEmotionStrategy(self.agent.emotion_state)
        self.agent.perception_accuracy = perception_accuracy
        self.agent.visited_states = defaultdict(int)
        self.agent.active_goals = []
        self.agent.working_memory_buffer = deque(maxlen=5)
        self.agent.attention_focus = None
        self.agent.internal_monologue = ""

        # Initialize the new modular components
        self.agent.perception_manager = PerceptionManager(self.agent)
        self.agent.action_executor = ActionExecutor(self.agent)
        self.agent.thought_processor = ThoughtProcessor(self.agent)

        # Initialize consciousness states
        self.agent.awake_state = AwakeState(self.agent)
        self.agent.asleep_state = AsleepState(self.agent)
        self.agent.focused_state = FocusedState(self.agent)
        self.agent.current_consciousness_state = self.agent.awake_state
        self.agent.current_consciousness_state.enter()

        self.agent.cognitive_modules = {}

    def initialize_goals(self):
        """
        Initializes a set of default goals for the agent, now supporting hierarchies.
        """
        main_goal_reach_center = {
            "id": "goal_reach_center",
            "type": "reach_location",
            "name": "Reach Center of Map",
            "priority": 0.9,
            "completed": False,
            "parent_goal_id": None,
            "prerequisites": [],
            "target_x": 5,
            "target_y": 5
        }
        self.agent.active_goals.append(main_goal_reach_center)

        clear_obstacle_sub_goal = {
            "id": "sub_goal_clear_obstacle",
            "type": "clear_path",
            "name": "Clear Obstacle on Path",
            "priority": 0.85,
            "completed": False,
            "parent_goal_id": "goal_reach_center",
            "prerequisites": [],
            "obstacle_location": None
        }
        self.agent.active_goals.append(clear_obstacle_sub_goal)

        self.agent.active_goals.append({
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

        self.agent.active_goals.append({
            "id": "goal_stay_hydrated",
            "type": "maintain_thirst_low",
            "name": "Stay Hydrated",
            "priority": 0.7,
            "completed": False,
            "parent_goal_id": None,
            "prerequisites": [],
            "threshold": 0.2,
            "duration_steps": 15,
            "current_duration": 0
        })

        print(f"{self.agent.name} initialized with goals: {[goal['name'] for goal in self.agent.active_goals]}")

    def initialize_cognitive_modules(self):
        """
        Initializes and registers plug-and-play cognitive modules.
        """
        problem_solver = ProblemSolver(self.agent)
        self.agent.cognitive_modules[problem_solver.get_module_name()] = problem_solver
        print(f"{self.agent.name} initialized with cognitive module: {problem_solver.get_module_name()}")

        goal_generator = GoalGenerator(self.agent)
        self.agent.cognitive_modules[goal_generator.get_module_name()] = goal_generator
        print(f"{self.agent.name} initialized with cognitive module: {goal_generator.get_module_name()}")

    def initialize_procedures(self):
        """
        Initializes a set of default procedures (learned skills/habits) for the agent.
        These procedures can be added dynamically or learned over time.
        """
        self.agent.procedural_memory.add_procedure(
            name="Emergency Food Search",
            condition={"type": "hunger_high", "threshold": 0.7},
            action_sequence=["seek_food"],
            priority=0.8,
            condition_description="When hunger is high"
        )

        self.agent.procedural_memory.add_procedure(
            name="Fatigue Recovery",
            condition={"type": "fatigue_high", "threshold": 0.7},
            action_sequence=["rest"],
            priority=0.7,
            condition_description="When fatigue is high"
        )

        self.agent.procedural_memory.add_procedure(
            name="Clear Obstacle",
            condition={"type": "obstacle_blocking_path"},
            action_sequence=["move_object"],
            priority=0.9,
            condition_description="When an obstacle is blocking a goal path"
        )

        self.agent.procedural_memory.add_procedure(
            name="Emergency Water Search",
            condition={"type": "thirst_high", "threshold": 0.7},
            action_sequence=["drink_water"],
            priority=0.85,
            condition_description="When thirst is high"
        )
        print(f"{self.agent.name} initialized with default procedures.")

    def initialize_semantic_memory(self):
        """
        Initializes the semantic memory with some default general knowledge facts.
        """
        self.agent.semantic_memory.add_fact("food",
                                            {"is_a": "resource", "property": "edible", "effect": "reduces_hunger"})
        self.agent.semantic_memory.add_fact("water",
                                            {"is_a": "resource", "property": "drinkable", "effect": "reduces_thirst"})
        self.agent.semantic_memory.add_fact("obstacle", {"is_a": "barrier", "property": "immovable_by_default",
                                                         "action_needed": "move_object"})
        self.agent.semantic_memory.add_fact("rest",
                                            {"is_a": "action", "effect": "reduces_fatigue", "context": "safe_place"})
        self.agent.semantic_memory.add_fact("explore", {"is_a": "action", "effect": "gains_information",
                                                        "cost": "increases_hunger_fatigue_thirst"})
        self.agent.semantic_memory.add_fact("shelter", {"is_a": "structure", "property": "provides_safety",
                                                        "context": "bad_weather"})
        print(f"{self.agent.name} initialized with default semantic facts.")

