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

from core.action.base_action_executor import BaseActionExecutor
from agent.base_agent import Agent


class ActionExecutor(BaseActionExecutor):
    """
    Manages the execution of an agent's actions and their consequences.

    This class encapsulates the logic for applying action effects on the agent's
    internal state, environment, and updating learning components (DQN, RewardLearner)
    and memory (EpisodicMemory).
    """

    def __init__(self, agent: Agent):
        """
        Initializes the ActionExecutor.

        Args:
            agent (Agent): The agent instance that this executor will perform actions for.
        """
        super().__init__(agent)
        # The agent object holds attributes like environment, internal_state,
        # last_action_reward, q_learner, episodic_memory, emotion_state,
        # short_term_memory, procedural_memory, reward_learner, pos_x, pos_y,
        # name, perception, current_time_step.
        # We will access these directly via self.agent.

    def execute_action(self, action: str):
        """
        Executes the chosen action and applies its effects on the agent's
        internal state, environment, and learning.

        Args:
            action (str): The action to perform.
        """
        # Take a snapshot of the agent's internal state before the action.
        previous_internal_state_snapshot = self.agent.internal_state.snapshot()
        original_pos_x, original_pos_y = self.agent.pos_x, self.agent.pos_y  # Store original position

        # Store the action that was *actually* performed, for procedural memory update
        performed_action_id = None
        # Check if the action was suggested by a procedure
        triggered_procedure = self.agent.procedural_memory.get_triggered_procedure(self.agent)
        if triggered_procedure and triggered_procedure["suggested_action"] == action:
            performed_action_id = triggered_procedure["id"]

        # Perform the selected action and apply its effects.
        if action == "seek_food":
            # Check if there's food at the agent's current location
            if self.agent.environment.grid[self.agent.pos_x][self.agent.pos_y] == 'food':
                self.agent.internal_state.hunger = max(0.0, self.agent.internal_state.hunger - 0.7)
                self.agent.last_action_reward = 0.5  # Positive reward for successfully finding food.
                self.agent.environment.grid[self.agent.pos_x][self.agent.pos_y] = 'empty'  # Consume food
                print(
                    f"{self.agent.name} successfully ate food at ({self.agent.pos_x},{self.agent.pos_y})! Hunger reduced.")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=True)
            else:
                self.agent.internal_state.hunger = min(1.0, self.agent.internal_state.hunger + 0.02)
                self.agent.last_action_reward = -0.1  # Small penalty for unsuccessful attempt.
                print(
                    f"{self.agent.name} sought food but found none at ({self.agent.pos_x},{self.agent.pos_y}). Hunger increased slightly.")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=False)
            # Update memory about food presence (if food was globally available or locally seen)
            if self.agent.perception["food_available_global"] or self.agent.perception["food_in_sight"]:
                self.agent.short_term_memory["food_last_seen"] = self.agent.current_time_step

        elif action == "drink_water":
            # Check if there's water at the agent's current location
            if self.agent.environment.grid[self.agent.pos_x][self.agent.pos_y] == 'water':
                self.agent.internal_state.thirst = max(0.0,
                                                       self.agent.internal_state.thirst - 0.8)  # Reduce thirst significantly
                self.agent.last_action_reward = 0.6  # Positive reward for successfully drinking water
                self.agent.environment.grid[self.agent.pos_x][self.agent.pos_y] = 'empty'  # Consume water
                print(
                    f"{self.agent.name} successfully drank water at ({self.agent.pos_x},{self.agent.pos_y})! Thirst reduced.")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=True)
            else:
                self.agent.internal_state.thirst = min(1.0,
                                                       self.agent.internal_state.thirst + 0.03)  # Small penalty for unsuccessful attempt
                self.agent.last_action_reward = -0.15  # Slightly higher penalty than food, as thirst increases faster
                print(
                    f"{self.agent.name} sought water but found none at ({self.agent.pos_x},{self.agent.pos_y}). Thirst increased slightly.")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=False)
            # Update memory about water presence
            if self.agent.perception["water_available_global"] or self.agent.perception["water_in_sight"]:
                self.agent.short_term_memory["water_last_seen"] = self.agent.current_time_step

        elif action == "rest":
            self.agent.internal_state.fatigue = max(0.0, self.agent.internal_state.fatigue - 0.6)
            self.agent.last_action_reward = 0.4  # Positive reward for resting.
            print(f"{self.agent.name} is resting at ({self.agent.pos_x},{self.agent.pos_y}). Fatigue reduced.")
            if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                          success=True)

        elif action == "explore":
            self.agent.internal_state.hunger = min(1.0, self.agent.internal_state.hunger + 0.05)
            self.agent.internal_state.fatigue = min(1.0, self.agent.internal_state.fatigue + 0.05)
            self.agent.internal_state.thirst = min(1.0,
                                                   self.agent.internal_state.thirst + 0.05)  # Exploration increases thirst
            self.agent.last_action_reward = -0.05
            print(
                f"{self.agent.name} is exploring at ({self.agent.pos_x},{self.agent.pos_y}). Hunger/Fatigue/Thirst increased slightly.")
            if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                          success=False)

        # --- Movement Actions ---
        elif action.startswith("move_"):
            new_pos_x, new_pos_y = self.agent.pos_x, self.agent.pos_y
            if action == "move_up":
                new_pos_x -= 1
            elif action == "move_down":
                new_pos_x += 1
            elif action == "move_left":
                new_pos_y -= 1
            elif action == "move_right":
                new_pos_y += 1

            # Check if new position is within grid boundaries AND not an obstacle
            grid_size = len(self.agent.environment.grid)
            if 0 <= new_pos_x < grid_size and 0 <= new_pos_y < grid_size:
                if self.agent.environment.grid[new_pos_x][new_pos_y] != 'obstacle':
                    self.agent.pos_x, self.agent.pos_y = new_pos_x, new_pos_y
                    self.agent.last_action_reward = -0.01  # Small cost for movement
                    print(
                        f"{self.agent.name} moved {action.replace('move_', '')} to ({self.agent.pos_x},{self.agent.pos_y}).")
                    if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                                  success=True)
                else:
                    self.agent.last_action_reward = -0.1  # Penalty for trying to move onto an obstacle
                    print(
                        f"{self.agent.name} tried to move {action.replace('move_', '')} onto an obstacle at ({new_pos_x},{new_pos_y}).")
                    if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                                  success=False)
            else:
                self.agent.last_action_reward = -0.1  # Penalty for trying to move out of bounds
                print(
                    f"{self.agent.name} tried to move {action.replace('move_', '')} out of bounds from ({original_pos_x},{original_pos_y}).")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=False)

        # --- Move Object Action ---
        elif action == "move_object":
            target_obj_x, target_obj_y = self.agent.pos_x, self.agent.pos_y

            possible_new_positions = [
                (target_obj_x, target_obj_y + 1),  # Right
                (target_obj_x + 1, target_obj_y),  # Down
                (target_obj_x, target_obj_y - 1),  # Left
                (target_obj_x - 1, target_obj_y)  # Up
            ]

            moved_successfully = False
            for new_obj_x, new_obj_y in possible_new_positions:
                if self.agent.environment.move_object_at_position(target_obj_x, target_obj_y, new_obj_x, new_obj_y):
                    self.agent.last_action_reward = 0.3  # Positive reward for moving an object
                    print(
                        f"{self.agent.name} successfully moved object from ({target_obj_x},{target_obj_y}) to ({new_obj_x},{new_obj_y}).")
                    moved_successfully = True
                    if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                                  success=True)
                    break

            if not moved_successfully:
                self.agent.last_action_reward = -0.2  # Penalty for failing to move object
                print(
                    f"{self.agent.name} tried to move an object at ({target_obj_x},{target_obj_y}) but failed (no object or no empty adjacent cell).")
                if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                              success=False)

        else:
            print(f"{self.agent.name} attempted an unknown action: {action}")
            self.agent.last_action_reward = -0.2  # Penalty for invalid action
            if performed_action_id: self.agent.procedural_memory.update_procedure_outcome(performed_action_id,
                                                                                          success=False)

        # Update the agent's internal state (hunger/fatigue/thirst naturally increase, mood recalculates)
        self.agent.internal_state.update(delta_time=1.0)

        # Update the simple RewardLearner
        adjusted_reward = self.agent.reward_learner.update(
            previous_internal_state_snapshot,
            self.agent.internal_state,
            action,
            self.agent.last_action_reward,
            self.agent.internal_state.mood_value
        )
        # Update the agent's last_action_reward with the adjusted value
        self.agent.last_action_reward = adjusted_reward

        # Update DQN with the experience using the adjusted reward
        self.agent.q_learner.update(
            previous_internal_state_snapshot['hunger'],
            previous_internal_state_snapshot['fatigue'],
            previous_internal_state_snapshot['thirst'],
            action,
            self.agent.last_action_reward,
            self.agent.internal_state.hunger, self.agent.internal_state.fatigue, self.agent.internal_state.thirst
        )

        # Store the episode in episodic memory with emotional weighting
        self.agent.episodic_memory.add(
            step=self.agent.current_time_step,
            perception=self.agent.perception,
            internal_state=self.agent.internal_state,
            action=action,
            emotions=self.agent.emotion_state
        )

        # Increment agent's internal step counter (for memory and other time-based checks)
        self.agent.current_time_step += 1

