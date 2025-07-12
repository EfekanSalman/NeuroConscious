# core/consciousness/focused_state.py

from core.consciousness.base import ConsciousnessState
import random

class FocusedState(ConsciousnessState):
    """
    Represents the agent's focused state of consciousness.

    In this state, the agent's perception accuracy towards its attention focus
    is significantly boosted, and its actions are more directly aimed at
    achieving the goal associated with that focus. Other distractions are reduced.
    """
    def enter(self):
        """
        Actions to perform when entering the Focused state.
        """
        print(f"{self.agent.name} is now Focused on {self.agent.attention_focus}.")
        self.agent.internal_monologue += f"I am entering a focused state, concentrating on {self.agent.attention_focus}. "


    def exit(self):
        """
        Actions to perform when exiting the Focused state.
        """
        print(f"{self.agent.name} is no longer Focused.")
        self.agent.internal_monologue += "My focus has shifted. "

    def sense(self):
        """
        Agent's sensing is enhanced towards its attention focus in the Focused state.
        Delegates to the agent's default sense method, but with a temporary boost
        to perception accuracy for the focused stimulus.
        """
        original_perception_accuracy = self.agent.perception_accuracy
        # Temporarily boost perception accuracy for the attention focus
        if self.agent.attention_focus == 'food':
            self.agent.perception_accuracy = min(1.0, original_perception_accuracy + 0.3) # Higher boost
        elif self.agent.attention_focus == 'location_target':
            self.agent.perception_accuracy = min(1.0, original_perception_accuracy + 0.3) # Higher boost
        # Other attention focuses can also get a boost

        self.agent._sense_default() # Call the agent's internal default sense method

        # Restore original perception accuracy after sensing
        self.agent.perception_accuracy = original_perception_accuracy
        self.agent.internal_monologue += f"In my focused state, I am acutely aware of {self.agent.attention_focus}. "


    def think(self) -> str:
        """
        Agent's thinking in the Focused state is heavily biased towards its attention focus.
        It will prioritize actions related to its focus, potentially overriding DQN's choice.
        """
        # First, let the default think process run to get initial action and update monologue
        initial_monologue_part = self.agent.internal_monologue # Store part before default think
        selected_action = self.agent._think_default() # Get action from default logic (DQN, goals, etc.)
        self.agent.internal_monologue = initial_monologue_part # Reset monologue to add focused thoughts

        # Now, apply focus-specific overrides
        if self.agent.attention_focus == 'food' and self.agent.internal_state.hunger > 0.3:
            if selected_action != "seek_food" and not selected_action.startswith("move_"):
                self.agent.internal_monologue += "My hunger is still a concern, and I am focused on food, so I must seek food. "
                selected_action = "seek_food"
            elif selected_action.startswith("move_"):
                # If already moving, ensure it's towards food if food is in working memory
                found_food_in_wm = False
                for item in reversed(self.agent.working_memory_buffer):
                    if item["type"] == "perceived_food" and self.agent.current_time_step - item["time"] <= 5:
                        food_x, food_y = item["location"]
                        if (self.agent.pos_x, self.agent.pos_y) != (food_x, food_y):
                            dist_x = abs(self.agent.pos_x - food_x)
                            dist_y = abs(self.agent.pos_y - food_y)
                            if dist_x > 0:
                                new_action = "move_down" if self.agent.pos_x < food_x else "move_up"
                            else:
                                new_action = "move_right" if self.agent.pos_y < food_y else "move_left"
                            if new_action != selected_action:
                                self.agent.internal_monologue += f"My focus on food directs me to move towards the recalled food at ({food_x},{food_y}). "
                                selected_action = new_action
                            found_food_in_wm = True
                            break
                if not found_food_in_wm and selected_action != "seek_food":
                     self.agent.internal_monologue += "Focused on food, but no immediate path. I will seek food generally. "
                     selected_action = "seek_food"


        elif self.agent.attention_focus == 'location_target':
            # Find the active 'reach_location' goal
            target_goal = next((g for g in self.agent.active_goals if g["type"] == "reach_location" and not g["completed"]), None)
            if target_goal:
                dist_x = abs(self.agent.pos_x - target_goal["target_x"])
                dist_y = abs(self.agent.pos_y - target_goal["target_y"])
                distance = dist_x + dist_y
                if distance > 0:
                    self.agent.internal_monologue += f"My focus on the target location ({target_goal['target_x']},{target_goal['target_y']}) makes me move directly towards it. "
                    if dist_x > 0:
                        selected_action = "move_down" if self.agent.pos_x < target_goal["target_x"] else "move_up"
                    else:
                        selected_action = "move_right" if self.agent.pos_y < target_goal["target_y"] else "move_left"
                else:
                    self.agent.internal_monologue += f"I have reached my target location for goal {target_goal['name']}! "
                    # The goal completion is handled in agent.think_default, so no need to duplicate here.

        # If no specific overrides, use the default action
        return selected_action

    def act(self, action: str):
        """
        Agent acts normally in the Focused state, but with potential for
        more precise movements or actions related to its focus.
        Delegates to the agent's default act method.

        Args:
            action (str): The action to perform.
        """
        self.agent._act_default(action) # Call the agent's internal default act method

    def get_state_name(self) -> str:
        """
        Returns the name of this consciousness state.
        """
        return "Focused"

