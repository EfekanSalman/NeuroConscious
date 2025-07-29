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

import random
from typing import List, Tuple, Dict, Any

from core.thought.base_thought_processor import BaseThoughtProcessor
from agent.base_agent import Agent


class ThoughtProcessor(BaseThoughtProcessor):
    """
    Manages the agent's complex thinking processes.

    This class encapsulates the logic for updating physiological and emotional states,
    setting attention focus, processing cognitive modules, and influencing action
    selection based on memory, weather, curiosity, and goals.
    It orchestrates the decision-making process by delegating to the DecisionMaker.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the ThoughtProcessor.

        Args:
            agent (Agent): The agent instance that this processor will manage thoughts for.
        """
        super().__init__(agent)
        # The agent object holds attributes like internal_state, emotion_strategy,
        # previous_physiological_states, motivation_engine, perception,
        # short_term_memory, emotion_state, attention_focus, internal_monologue,
        # cognitive_modules, active_goals, procedural_memory, semantic_memory,
        # q_learner, visited_states, current_time_step, pos_x, pos_y,
        # last_action_reward, working_memory_buffer, decision_maker, _last_performed_action.
        # We will access these directly via self.agent.

    def process_thought(self, decision_mode: str) -> str:
        """
        Processes the agent's internal state and perceptions to decide on an action.

        This method orchestrates the agent's thinking process, including:
        - Updating physiological and emotional states.
        - Setting attention focus based on needs and goals.
        - Processing outputs from cognitive modules (e.g., GoalGenerator).
        - Influencing decisions based on procedural and semantic memory.
        - Selecting an action using the DQN and applying various overrides
          based on memory, environment (weather, other agents), curiosity, and goals.
        - Generating an internal monologue to trace reasoning.

        Args:
            decision_mode (str): The current decision-making mode ('reactive' or 'deliberative').

        Returns:
            str: The chosen action.
        """
        # 1. Update physiological state (e.g., hunger, fatigue, thirst increases over time).
        time_delta_factor = 1.5 if self.agent.perception["time_of_day"] == "night" else 1.0
        self.agent.internal_state.update(delta_time=time_delta_factor)

        # 2. Update emotional state based on environmental perceptions and internal state.
        self.agent.emotion_strategy.update_emotions(self.agent.perception, self.agent.internal_state)

        # 3. Store current state before acting (for learning)
        self.agent.previous_physiological_states = (self.agent.internal_state.hunger, self.agent.internal_state.fatigue,
                                                    self.agent.internal_state.thirst)

        # 4. Emotion-driven motivation engine decides preferred action (not directly used for final action here)
        _ = self.agent.motivation_engine.decide_action(
            perception=self.agent.perception,
            memory=self.agent.short_term_memory,
            emotions=self.agent.emotion_state
        )

        # --- Set Attention Focus ---
        # Prioritize attention based on most pressing needs or active goals
        if self.agent.internal_state.hunger > 0.7:
            self.agent.attention_focus = 'food'
            self.agent.internal_monologue += "I am very hungry, so I should focus on finding food. "
        elif self.agent.internal_state.thirst > 0.7:
            self.agent.attention_focus = 'water'
            self.agent.internal_monologue += "I am very thirsty, so I should focus on finding water. "
        elif self.agent.internal_state.fatigue > 0.7:
            self.agent.attention_focus = 'rest'
            self.agent.internal_monologue += "I am very fatigued, so I should focus on resting. "
        elif self.agent.emotion_state.get("curiosity") > 0.6 and not self.agent.active_goals:
            self.agent.attention_focus = 'curiosity'
            self.agent.internal_monologue += "I feel curious and have no immediate goals, so I will focus on exploration. "
        elif any(not goal["completed"] for goal in self.agent.active_goals):
            uncompleted_goals = [g for g in self.agent.active_goals if not g["completed"]]
            if uncompleted_goals:
                highest_priority_goal = max(uncompleted_goals, key=lambda g: g["priority"])
                if highest_priority_goal["type"] == "reach_location":
                    self.agent.attention_focus = 'location_target'
                    self.agent.internal_monologue += f"My top priority is to reach {highest_priority_goal['name']} at ({highest_priority_goal['target_x']},{highest_priority_goal['target_y']}). "
                elif highest_priority_goal["type"] == "maintain_hunger_low":
                    self.agent.attention_focus = 'food'
                    self.agent.internal_monologue += f"I need to maintain low hunger for {highest_priority_goal['name']}, so I will focus on food. "
                elif highest_priority_goal["type"] == "maintain_thirst_low":
                    self.agent.attention_focus = 'water'
                    self.agent.internal_monologue += f"I need to maintain low thirst for {highest_priority_goal['name']}, so I will focus on water. "
                elif highest_priority_goal["type"] == "clear_path":
                    self.agent.attention_focus = 'obstacle'
                    self.agent.internal_monologue += f"My top priority is to clear a path for {highest_priority_goal['name']}. "
                elif highest_priority_goal["type"] == "explore_area":
                    self.agent.attention_focus = 'curiosity'
                    self.agent.internal_monologue += f"My top priority is to explore new areas for {highest_priority_goal['name']}. "
                else:
                    self.agent.attention_focus = None
                    self.agent.internal_monologue += "I have an active goal but no specific focus for it. "
            else:
                self.agent.attention_focus = None
                self.agent.internal_monologue += "No active goals to focus on. "
        elif self.agent.perception["obstacle_in_sight"]:
            self.agent.attention_focus = 'obstacle'
            self.agent.internal_monologue += "I see an obstacle, focusing on how to deal with it. "
        else:
            self.agent.attention_focus = None
            self.agent.internal_monologue += "No specific attention focus right now. "

        # --- Process Cognitive Modules ---
        cognitive_module_outputs: Dict[str, Any] = {}
        for module_name, module_instance in self.agent.cognitive_modules.items():
            module_output = module_instance.process()
            if module_output:
                cognitive_module_outputs[module_name] = module_output
                self.agent.internal_monologue += f"Cognitive module '{module_name}' provided output: {module_output}. "

        # Process GoalGenerator output
        if "GoalGenerator" in self.agent.cognitive_modules:
            goal_generator_output = cognitive_module_outputs.get("GoalGenerator", {})
            if "new_goals" in goal_generator_output:
                for new_goal in goal_generator_output["new_goals"]:
                    if not any(g["id"] == new_goal["id"] for g in self.agent.active_goals):
                        self.agent.active_goals.append(new_goal)
                        self.agent.internal_monologue += f"GoalGenerator: Added new goal '{new_goal['name']}'. "
            if "modify_goals" in goal_generator_output:
                for modify_instruction in goal_generator_output["modify_goals"]:
                    goal_id_to_modify = modify_instruction["id"]
                    for goal in self.agent.active_goals:
                        if goal["id"] == goal_id_to_modify:
                            for key, value in modify_instruction.items():
                                if key != "id":
                                    goal[key] = value
                            self.agent.internal_monologue += f"GoalGenerator: Modified goal '{goal['name']}'. "
                            break

        # --- Procedural Memory Influence ---
        triggered_procedure = self.agent.procedural_memory.get_triggered_procedure(self.agent)
        if triggered_procedure:
            self.agent.internal_monologue += f"Procedural Memory suggests: '{triggered_procedure['name']}' ({triggered_procedure['suggested_action']}). "

        # --- Semantic Memory Influence ---
        if self.agent.internal_state.hunger > 0.6:
            food_facts = self.agent.semantic_memory.retrieve_facts("food")
            if food_facts:
                self.agent.internal_monologue += f"Semantic Memory reminds me that 'food' is {food_facts.get('property', 'unknown')} and {food_facts.get('effect', 'has no effect')}. "
                if self.agent.perception["food_in_sight"] and self.agent.semantic_memory.infer_property("food",
                                                                                                        "effect") == "reduces_hunger":
                    self.agent.internal_monologue += "Food in sight and known to reduce hunger, considering 'seek_food'. "

        if self.agent.internal_state.thirst > 0.6:
            water_facts = self.agent.semantic_memory.retrieve_facts("water")
            if water_facts:
                self.agent.internal_monologue += f"Semantic Memory reminds me that 'water' is {water_facts.get('property', 'unknown')} and {water_facts.get('effect', 'has no effect')}. "
                if self.agent.perception["water_in_sight"] and self.agent.semantic_memory.infer_property("water",
                                                                                                         "effect") == "reduces_thirst":
                    self.agent.internal_monologue += "Water in sight and known to reduce thirst, considering 'drink_water'. "

        if self.agent.internal_state.fatigue > 0.6:
            rest_facts = self.agent.semantic_memory.retrieve_facts("rest")
            if rest_facts and rest_facts.get("effect") == "reduces_fatigue":
                self.agent.internal_monologue += f"Semantic Memory reminds me that 'rest' {rest_facts.get('effect', 'has no effect')}. "

        current_weather = self.agent.perception["current_weather"]
        if current_weather == "stormy":
            shelter_facts = self.agent.semantic_memory.retrieve_facts("shelter")
            if shelter_facts and shelter_facts.get("property") == "provides_safety" and shelter_facts.get(
                    "context") == "bad_weather":
                self.agent.internal_monologue += f"Semantic Memory informs me that 'shelter' provides safety in bad weather. "

        # --- Decision Maker determines the final action based on the mode ---
        final_action = self.agent.decision_maker.decide_final_action(self.agent, decision_mode)

        self.agent.internal_monologue += f"Therefore, I have decided to {final_action}. "
        self.agent._last_performed_action = final_action
        return final_action

