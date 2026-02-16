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
from collections import deque  # For working memory
from core.perception.base_perception_manager import BasePerceptionManager
from agent.base_agent import Agent


class PerceptionManager(BasePerceptionManager):
    """
    Manages the agent's perception of its environment.

    This class encapsulates the logic for sensing the global environment,
    local grid view, applying sensory noise, and updating the agent's
    perception dictionary and working memory.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the PerceptionManager.

        Args:
            agent (Agent): The agent instance that this manager will update perceptions for.
        """
        super().__init__(agent)
        # The agent object holds attributes like environment, perception,
        # short_term_memory, current_time_step, perception_accuracy,
        # attention_focus, and working_memory_buffer.
        # We will access these directly via self.agent.

    def update_perception(self):
        """
        Updates the agent's perceptions based on the current environment state and local grid view.

        This method reads global information (food availability, time of day, weather) from the World
        and also scans the immediate surroundings (e.g., 1-cell radius) on the grid for items like food,
        water, obstacles, and other agents. Sensory noise is applied, meaning perceptions might be imperfect.
        Important perceptions are also added to the working memory buffer.
        An attention system can modify perception accuracy for specific stimuli.
        """
        if self.agent.environment:
            # Global perceptions
            self.agent.perception["food_available_global"] = self.agent.environment.food_available
            self.agent.perception["water_available_global"] = self.agent.environment.water_available
            self.agent.perception["time_of_day"] = self.agent.environment.time_of_day
            self.agent.perception["current_weather"] = self.agent.environment.current_weather
            self.agent.current_time_step = self.agent.environment.time_step

            # Local grid perception (e.g., 1-cell radius around the agent)
            raw_local_grid_view = self._get_local_grid_view(radius=1)

            # Apply sensory noise to local grid view with attention modulation
            self.agent.perception["local_grid_view"] = []
            self.agent.perception["food_in_sight"] = False
            self.agent.perception["water_in_sight"] = False
            self.agent.perception["water_locations"] = []
            self.agent.perception["obstacle_in_sight"] = False
            self.agent.perception["obstacle_locations"] = []

            for r_idx, row_content in enumerate(raw_local_grid_view):
                processed_row = []
                for c_idx, cell_content in enumerate(row_content):
                    # Adjust perception accuracy based on attention focus
                    current_perception_accuracy = self.agent.perception_accuracy
                    if self.agent.attention_focus == 'food' and cell_content == 'food':
                        current_perception_accuracy = min(1.0, self.agent.perception_accuracy + 0.2)
                    elif self.agent.attention_focus == 'water' and cell_content == 'water':
                        current_perception_accuracy = min(1.0, self.agent.perception_accuracy + 0.2)
                    elif self.agent.attention_focus == 'other_agents' and cell_content == 'agent':
                        current_perception_accuracy = min(1.0, self.agent.perception_accuracy + 0.2)
                    elif self.agent.attention_focus == 'obstacle' and cell_content == 'obstacle':
                        current_perception_accuracy = min(1.0, self.agent.perception_accuracy + 0.2)

                    if random.random() < current_perception_accuracy:
                        processed_row.append(cell_content)
                        if cell_content == 'food':
                            self.agent.perception["food_in_sight"] = True
                            abs_r = self.agent.pos_x + (r_idx - 1)
                            abs_c = self.agent.pos_y + (c_idx - 1)
                            self.agent.working_memory_buffer.append(
                                {"type": "perceived_food", "location": (abs_r, abs_c),
                                 "time": self.agent.current_time_step})
                        elif cell_content == 'water':
                            self.agent.perception["water_in_sight"] = True
                            abs_r = self.agent.pos_x + (r_idx - 1)
                            abs_c = self.agent.pos_y + (c_idx - 1)
                            self.agent.perception["water_locations"].append((abs_r, abs_c))
                            self.agent.working_memory_buffer.append(
                                {"type": "perceived_water", "location": (abs_r, abs_c),
                                 "time": self.agent.current_time_step})
                        elif cell_content == 'obstacle':
                            self.agent.perception["obstacle_in_sight"] = True
                            abs_r = self.agent.pos_x + (r_idx - 1)
                            abs_c = self.agent.pos_y + (c_idx - 1)
                            self.agent.perception["obstacle_locations"].append((abs_r, abs_c))
                            self.agent.working_memory_buffer.append(
                                {"type": "perceived_obstacle", "location": (abs_r, abs_c),
                                 "time": self.agent.current_time_step})
                    else:
                        processed_row.append('unknown')
                self.agent.perception["local_grid_view"].append(processed_row)

            # Check for other agents in local view (also apply noise with attention modulation)
            self.agent.perception["other_agents_in_sight"] = []
            grid_size = len(self.agent.environment.grid)
            radius = 1

            for r_offset in range(-radius, radius + 1):
                for c_offset in range(-radius, radius + 1):
                    view_row, view_col = self.agent.pos_x + r_offset, self.agent.pos_y + c_offset

                    if view_row == self.agent.pos_x and view_col == self.agent.pos_y:
                        continue

                    if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                        current_agent_perception_accuracy = self.agent.perception_accuracy
                        if self.agent.attention_focus == 'other_agents':
                            current_agent_perception_accuracy = min(1.0, self.agent.perception_accuracy + 0.2)

                        if random.random() < current_agent_perception_accuracy:
                            for other_agent in self.agent.environment.agents:
                                if other_agent is not self.agent and other_agent.pos_x == view_row and other_agent.pos_y == view_col:
                                    agent_info = {"name": other_agent.name, "pos_x": other_agent.pos_x,
                                                  "pos_y": other_agent.pos_y}
                                    self.agent.perception["other_agents_in_sight"].append(agent_info)
                                    self.agent.working_memory_buffer.append(
                                        {"type": "perceived_agent", "info": agent_info,
                                         "time": self.agent.current_time_step})
                                    break

            if self.agent.perception["food_available_global"] or self.agent.perception["food_in_sight"]:
                self.agent.short_term_memory["food_last_seen"] = self.agent.current_time_step

            if self.agent.perception["water_available_global"] or self.agent.perception["water_in_sight"]:
                self.agent.short_term_memory["water_last_seen"] = self.agent.current_time_step

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
        grid_size = len(self.agent.environment.grid)

        for r_offset in range(-radius, radius + 1):
            row_view = []
            for c_offset in range(-radius, radius + 1):
                view_row, view_col = self.agent.pos_x + r_offset, self.agent.pos_y + c_offset
                if 0 <= view_row < grid_size and 0 <= view_col < grid_size:
                    row_view.append(self.agent.environment.grid[view_row][view_col])
                else:
                    row_view.append('boundary')
            view.append(row_view)
        return view

