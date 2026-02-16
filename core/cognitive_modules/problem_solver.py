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

from core.cognitive_modules.base_module import CognitiveModule
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from agent.base_agent import Agent


class ProblemSolver(CognitiveModule):
    """
    An example cognitive module for basic problem-solving, focused on goals.

    This module attempts to identify uncompleted goals and suggests actions
    to move towards their completion.
    """

    def __init__(self, agent: 'Agent'):
        super().__init__(agent, name="ProblemSolver")
        self.last_goal_check_step = -1

    def process(self) -> Dict[str, Any]:
        """
        Analyzes active goals and suggests a course of action to solve them.

        Returns:
            Dict[str, Any]: A dictionary that might contain 'suggested_action'
                            or other influences.
        """
        output = {}

        # Only process every few steps to simulate complex thought taking time
        if self.agent.current_time_step - self.last_goal_check_step < 5:
            return output
        self.last_goal_check_step = self.agent.current_time_step

        uncompleted_goals = [g for g in self.agent.active_goals if not g["completed"]]

        if uncompleted_goals:
            # Prioritize the highest priority uncompleted goal
            current_goal = max(uncompleted_goals, key=lambda g: g["priority"])

            self.agent.internal_monologue += f"ProblemSolver: Analyzing goal '{current_goal['name']}'. "

            if current_goal["type"] == "reach_location":
                target_x, target_y = current_goal["target_x"], current_goal["target_y"]
                current_x, current_y = self.agent.pos_x, self.agent.pos_y

                # If already at target, mark as completed (should be handled by DecisionMaker too)
                if current_x == target_x and current_y == target_y:
                    current_goal["completed"] = True
                    self.agent.internal_monologue += f"ProblemSolver: Goal '{current_goal['name']}' achieved. "
                    return output  # No action needed from here

                # Suggest movement towards the target
                if abs(current_x - target_x) > abs(current_y - target_y):
                    suggested_action = "move_down" if current_x < target_x else "move_up"
                else:
                    suggested_action = "move_right" if current_y < target_y else "move_left"

                output["suggested_action"] = suggested_action
                self.agent.internal_monologue += f"ProblemSolver: Suggesting '{suggested_action}' to reach goal. "

            elif current_goal["type"] == "maintain_hunger_low":
                if self.agent.internal_state.hunger >= current_goal["threshold"] * 0.8:  # If hunger is getting high
                    output["suggested_action"] = "seek_food"
                    self.agent.internal_monologue += f"ProblemSolver: Hunger is rising for '{current_goal['name']}', suggesting 'seek_food'. "

        return output

