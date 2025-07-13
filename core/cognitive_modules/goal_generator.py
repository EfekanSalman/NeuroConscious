# core/cognitive_modules/goal_generator.py

import random
from core.cognitive_modules.base_module import CognitiveModule
from typing import TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    from agent.base_agent import Agent


class GoalGenerator(CognitiveModule):
    """
    A cognitive module responsible for dynamically generating or modifying goals
    based on the agent's internal state, perception, and existing goals.

    This module helps the agent to be more proactive in setting objectives.
    """

    def __init__(self, agent: 'Agent'):
        super().__init__(agent, name="GoalGenerator")
        self.last_generation_step = -1
        self.generation_cooldown = 10  # Cooldown steps before generating new goals

    def process(self) -> Dict[str, Any]:
        """
        Processes information to suggest new goals or modifications to existing ones.

        Returns:
            Dict[str, Any]: A dictionary containing suggestions for new goals or
                            modifications, e.g., {"new_goals": [...], "modify_goals": [...]}.
        """
        output = {"new_goals": [], "modify_goals": []}

        # Apply a cooldown to prevent excessive goal generation
        if self.agent.current_time_step - self.last_generation_step < self.generation_cooldown:
            return output

        self.last_generation_step = self.agent.current_time_step

        self.agent.internal_monologue += "GoalGenerator: Evaluating potential new goals. "

        # --- Rule 1: Generate 'seek_food' goal if hunger is high and no active food goal ---
        if self.agent.internal_state.hunger > 0.6:
            has_seek_food_goal = any(
                g["type"] == "maintain_hunger_low" and not g["completed"] for g in self.agent.active_goals)
            if not has_seek_food_goal:
                new_food_goal = {
                    "id": f"goal_seek_food_{self.agent.current_time_step}",
                    "type": "maintain_hunger_low",
                    "name": "Satisfy Hunger",
                    "priority": 0.75,
                    "completed": False,
                    "parent_goal_id": None,
                    "prerequisites": [],
                    "threshold": 0.3,
                    "duration_steps": 5,  # Short duration for immediate hunger
                    "current_duration": 0
                }
                output["new_goals"].append(new_food_goal)
                self.agent.internal_monologue += f"GoalGenerator: Generated new goal 'Satisfy Hunger' due to high hunger. "

        # --- Rule 2: Generate 'explore' goal if curiosity is high and no active exploration/location goal ---
        if self.agent.emotion_state.get("curiosity") > 0.7 and \
                not any(g["type"] == "reach_location" and not g["completed"] for g in self.agent.active_goals) and \
                not any(g["type"] == "explore_area" and not g["completed"] for g in
                        self.agent.active_goals):  # Assuming a future 'explore_area' type

            # Simple exploration goal: reach a random unvisited area (conceptually)
            # For now, let's make it a low priority 'reach_location' to a random spot
            target_x = random.randint(0, self.agent.environment.GRID_SIZE - 1)
            target_y = random.randint(0, self.agent.environment.GRID_SIZE - 1)

            new_explore_goal = {
                "id": f"goal_explore_{self.agent.current_time_step}",
                "type": "reach_location",
                "name": f"Explore Area ({target_x},{target_y})",
                "priority": 0.3,  # Lower priority
                "completed": False,
                "parent_goal_id": None,
                "prerequisites": [],
                "target_x": target_x,
                "target_y": target_y
            }
            output["new_goals"].append(new_explore_goal)
            self.agent.internal_monologue += f"GoalGenerator: Generated new goal 'Explore Area' due to high curiosity. "

        # --- Rule 3: Modify 'clear_path' sub-goal if obstacle_location is cleared ---
        clear_path_goal = next(
            (g for g in self.agent.active_goals if g["id"] == "sub_goal_clear_obstacle" and not g["completed"]), None)
        if clear_path_goal and clear_path_goal["obstacle_location"]:
            obs_x, obs_y = clear_path_goal["obstacle_location"]
            # Check if the obstacle at the recorded location is no longer there
            if self.agent.environment.grid[obs_x][obs_y] != 'obstacle':
                output["modify_goals"].append({
                    "id": "sub_goal_clear_obstacle",
                    "completed": True,  # Mark as completed
                    "reason": "Obstacle removed"
                })
                self.agent.internal_monologue += f"GoalGenerator: Marked 'Clear Obstacle' sub-goal as completed because obstacle at ({obs_x},{obs_y}) is gone. "

        return output

