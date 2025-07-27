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
from typing import Dict, Any, Tuple


class DecisionMaker:
    """
    Manages the agent's high-level decision-making process, acting as a central
    arbiter for action selection.

    It takes inputs from various cognitive modules (DQN, needs, goals, memories)
    and applies a hierarchical logic or set of rules to determine the final action.
    This version incorporates deliberative and reactive decision-making modes,
    includes logic for moving objects, supports complex goal hierarchies
    with prerequisites and sub-goals, integrates with the GoalGenerator,
    and now leverages semantic memory for more informed decisions.
    """

    def __init__(self):
        """
        Initializes the DecisionMaker.
        """
        pass

    def decide_final_action(self, agent, decision_mode: str) -> str:
        """
        Determines the final action for the agent based on a hierarchy of considerations
        and the current decision-making mode (deliberative or reactive).

        Args:
            agent: The agent instance, providing access to its internal state,
                   perception, goals, and outputs from other cognitive modules.
            decision_mode (str): The current decision-making mode ('reactive' or 'deliberative').

        Returns:
            str: The final chosen action.
        """
        chosen_action = None
        agent.internal_monologue += f"Current decision mode: {decision_mode.capitalize()}. "

        # --- Decision Hierarchy ---

        # 1. Critical Needs Override (Highest Priority in both modes, but more immediate in Reactive)
        if agent.internal_state.hunger > 0.85:
            chosen_action = "seek_food"
            agent.internal_monologue += "CRITICAL: Extreme hunger detected, prioritizing 'seek_food'. "
            return chosen_action  # Override and return immediately

        if agent.internal_state.thirst > 0.85:  # New: Critical thirst override
            chosen_action = "drink_water"
            agent.internal_monologue += "CRITICAL: Extreme thirst detected, prioritizing 'drink_water'. "
            return chosen_action  # Override and return immediately

        if agent.internal_state.fatigue > 0.85:
            chosen_action = "rest"
            agent.internal_monologue += "CRITICAL: Extreme fatigue detected, prioritizing 'rest'. "
            return chosen_action  # Override and return immediately

        # 2. Procedural Memory Override (Stronger in Reactive Mode)
        triggered_procedure = agent.procedural_memory.get_triggered_procedure(agent)
        if triggered_procedure:
            # Reactive mode gives higher weight to procedural memory
            proc_priority_threshold = 0.7 if decision_mode == 'reactive' else 0.9  # Reactive acts on lower priority procedures
            if triggered_procedure["priority"] >= proc_priority_threshold:
                proc_action = triggered_procedure["suggested_action"]
                agent.internal_monologue += f"Procedural memory triggered (Priority: {triggered_procedure['priority']:.2f}): '{triggered_procedure['condition_description']}' suggests '{proc_action}'. "

                if proc_action == "move_towards_goal":
                    target_goal = next(
                        (g for g in agent.active_goals if g["type"] == "reach_location" and not g["completed"]), None)
                    if target_goal:
                        dist_x = abs(agent.pos_x - target_goal["target_x"])
                        dist_y = abs(agent.pos_y - target_goal["target_y"])
                        if dist_x > 0:
                            chosen_action = "move_down" if agent.pos_x < target_goal["target_x"] else "move_up"
                        elif dist_y > 0:
                            chosen_action = "move_right" if agent.pos_y < target_goal["target_y"] else "move_left"
                        else:
                            chosen_action = "explore"  # Already at goal, explore or find new goal
                        agent.internal_monologue += f"Following procedure, moving towards goal: {chosen_action}. "
                    else:
                        agent.internal_monologue += "Procedural 'move_towards_goal' triggered but no active location goal. Defaulting to explore. "
                        chosen_action = "explore"
                else:
                    chosen_action = proc_action
                    agent.internal_monologue += f"Following procedure, overriding to {chosen_action}. "
                return chosen_action  # Override and return immediately if procedure is strong enough

        # If no critical needs or strong procedural overrides, proceed with other logic.
        # Default to DQN suggestion if nothing else overrides it.
        initial_dqn_suggestion = agent.q_learner.choose_action(
            hunger=agent.internal_state.hunger,
            fatigue=agent.internal_state.fatigue,
            thirst=agent.internal_state.thirst
        )
        chosen_action = initial_dqn_suggestion
        agent.internal_monologue += f"Defaulting to DQN suggestion: {chosen_action}. "

        # --- Complex Goal System Influence (More prominent in Deliberative Mode) ---
        # Prioritize goals based on their priority, prerequisites, and parent-child relationships.
        # We need to find the most relevant uncompleted goal.
        relevant_goal = None
        highest_priority = -1.0

        # Filter and sort goals: first by completion status, then by prerequisites, then by priority
        uncompleted_goals = [g for g in agent.active_goals if not g["completed"]]

        # Sort goals: Higher priority first, then by whether prerequisites are met
        # For simplicity, we'll iterate and find the most relevant one based on a simple heuristic.
        # A more robust solution would involve a proper planning algorithm.
        for goal in uncompleted_goals:
            # Check prerequisites: A goal is only considered if its prerequisites are met.
            prerequisites_met = True
            for prereq_id in goal.get("prerequisites", []):
                prereq_goal = next((g for g in agent.active_goals if g["id"] == prereq_id), None)
                if prereq_goal and not prereq_goal["completed"]:
                    prerequisites_met = False
                    break

            if not prerequisites_met:
                agent.internal_monologue += f"Goal '{goal['name']}' has unmet prerequisites. Skipping for now. "
                continue

            # Consider parent-child relationships: If a parent goal is active, its sub-goals might get a boost.
            effective_priority = goal["priority"]
            if goal.get("parent_goal_id"):
                parent_goal = next((g for g in agent.active_goals if g["id"] == goal["parent_goal_id"]), None)
                if parent_goal and not parent_goal["completed"]:
                    # Boost sub-goal priority if parent is active
                    effective_priority = min(1.0, effective_priority + (parent_goal["priority"] * 0.1))  # Small boost

            if effective_priority > highest_priority:
                highest_priority = effective_priority
                relevant_goal = goal

        if relevant_goal and decision_mode == 'deliberative':
            agent.internal_monologue += f"DELIBERATIVE: Focusing on highest priority goal: '{relevant_goal['name']}' (Effective Priority: {highest_priority:.2f}). "

            if relevant_goal["type"] == "reach_location":
                target_x, target_y = relevant_goal["target_x"], relevant_goal["target_y"]
                dist_x = abs(agent.pos_x - target_x)
                dist_y = abs(agent.pos_y - target_y)
                distance = dist_x + dist_y

                if distance == 0:
                    relevant_goal["completed"] = True
                    agent.internal_monologue += f"Goal completed: {relevant_goal['name']}! "
                    # Consider finding a new goal or reverting to exploration
                    chosen_action = "explore"  # Fallback
                else:
                    # Check for obstacles blocking the path to the goal
                    blocking_obstacle_loc = None
                    if agent.perception["obstacle_in_sight"]:
                        # Simple check: Is there an obstacle between agent and target?
                        # This is a very basic check, a real pathfinding would be needed for complex maps.
                        for obs_loc in agent.perception["obstacle_locations"]:
                            # If obstacle is on the direct path (simplistic check)
                            if (min(agent.pos_x, target_x) <= obs_loc[0] <= max(agent.pos_x, target_x) and
                                    min(agent.pos_y, target_y) <= obs_loc[1] <= max(agent.pos_y, target_y)):
                                blocking_obstacle_loc = obs_loc
                                break

                    # If an obstacle is blocking the path, activate/prioritize the 'clear_path' sub-goal
                    clear_path_goal = next((g for g in agent.active_goals if g["id"] == "sub_goal_clear_obstacle"),
                                           None)
                    if blocking_obstacle_loc and clear_path_goal:
                        clear_path_goal["obstacle_location"] = blocking_obstacle_loc
                        clear_path_goal["completed"] = False  # Ensure it's active
                        agent.internal_monologue += f"DELIBERATIVE: Obstacle at {blocking_obstacle_loc} blocking path to '{relevant_goal['name']}'. Prioritizing 'clear_path' sub-goal. "
                        # The ProblemSolver (if active) or DecisionMaker itself should now pick 'move_object'
                        # For now, we'll directly suggest move_object if agent is adjacent to the obstacle.
                        if (abs(agent.pos_x - blocking_obstacle_loc[0]) <= 1 and
                                abs(agent.pos_y - blocking_obstacle_loc[1]) <= 1):
                            chosen_action = "move_object"
                            agent.internal_monologue += "Agent is adjacent to blocking obstacle, attempting to move object. "
                        else:
                            # Move towards the obstacle to clear it
                            if abs(agent.pos_x - blocking_obstacle_loc[0]) > abs(
                                    agent.pos_y - blocking_obstacle_loc[1]):
                                chosen_action = "move_down" if agent.pos_x < blocking_obstacle_loc[0] else "move_up"
                            else:
                                chosen_action = "move_right" if agent.pos_y < blocking_obstacle_loc[1] else "move_left"
                            agent.internal_monologue += f"Moving towards obstacle at {blocking_obstacle_loc} to clear path: {chosen_action}. "
                    else:
                        # No blocking obstacle or clear_path goal not defined/relevant, continue moving towards location
                        if dist_x > 0:
                            chosen_action = "move_down" if agent.pos_x < target_x else "move_up"
                        elif dist_y > 0:
                            chosen_action = "move_right" if agent.pos_y < target_y else "move_left"
                        agent.internal_monologue += f"Moving towards goal '{relevant_goal['name']}': {chosen_action}. "

            elif relevant_goal["type"] == "maintain_hunger_low":
                if agent.internal_state.hunger < relevant_goal["threshold"]:
                    relevant_goal["current_duration"] += 1
                    if relevant_goal["current_duration"] >= relevant_goal["duration_steps"]:
                        relevant_goal["completed"] = True
                        agent.internal_monologue += f"DELIBERATIVE: Goal completed: {relevant_goal['name']} (maintained hunger)! "
                    if agent.internal_state.hunger >= relevant_goal["threshold"] * 0.8 and chosen_action != "seek_food":
                        agent.internal_monologue += f"DELIBERATIVE: Hunger approaching threshold for goal '{relevant_goal['name']}', suggesting 'seek_food'. "
                        chosen_action = "seek_food"
                else:
                    relevant_goal["current_duration"] = 0

            elif relevant_goal["type"] == "maintain_thirst_low":  # New: Handle maintain_thirst_low goal
                if agent.internal_state.thirst < relevant_goal["threshold"]:
                    relevant_goal["current_duration"] += 1
                    if relevant_goal["current_duration"] >= relevant_goal["duration_steps"]:
                        relevant_goal["completed"] = True
                        agent.internal_monologue += f"DELIBERATIVE: Goal completed: {relevant_goal['name']} (maintained thirst)! "
                    if agent.internal_state.thirst >= relevant_goal[
                        "threshold"] * 0.8 and chosen_action != "drink_water":
                        agent.internal_monologue += f"DELIBERATIVE: Thirst approaching threshold for goal '{relevant_goal['name']}', suggesting 'drink_water'. "
                        chosen_action = "drink_water"
                else:
                    relevant_goal["current_duration"] = 0

            elif relevant_goal["type"] == "clear_path":
                # If this sub-goal is active, its obstacle_location should be set.
                if relevant_goal["obstacle_location"]:
                    obs_x, obs_y = relevant_goal["obstacle_location"]
                    # Check if the obstacle is still at the location (it might have been moved by another agent or disappeared)
                    if agent.environment.grid[obs_x][obs_y] != 'obstacle':
                        relevant_goal["completed"] = True
                        agent.internal_monologue += f"DELIBERATIVE: Obstacle at ({obs_x},{obs_y}) is gone. 'Clear Path' goal completed. "
                        # Fallback to parent goal or exploration
                        chosen_action = "explore"
                    elif agent.pos_x == obs_x and agent.pos_y == obs_y:
                        chosen_action = "move_object"
                        agent.internal_monologue += f"DELIBERATIVE: At obstacle location ({obs_x},{obs_y}), attempting to move object. "
                    else:
                        # Move towards the obstacle
                        if abs(agent.pos_x - obs_x) > abs(agent.pos_y - obs_y):
                            chosen_action = "move_down" if agent.pos_x < obs_x else "move_up"
                        else:
                            chosen_action = "move_right" if agent.pos_y < obs_y else "move_left"
                        agent.internal_monologue += f"Moving towards obstacle at ({obs_x},{obs_y}) to clear path: {chosen_action}. "
                else:
                    # If clear_path goal is active but no obstacle_location, it means it's waiting for an obstacle to appear or be identified.
                    agent.internal_monologue += "DELIBERATIVE: Clear path goal active but no specific obstacle identified yet. "
                    # Fallback to exploration or parent goal logic
                    chosen_action = "explore"

            elif relevant_goal["type"] == "explore_area":  # Handle explore_area goal
                target_x, target_y = relevant_goal["target_x"], relevant_goal["target_y"]
                dist_x = abs(agent.pos_x - target_x)
                dist_y = abs(agent.pos_y - target_y)
                distance = dist_x + dist_y

                if distance == 0:
                    relevant_goal["completed"] = True
                    agent.internal_monologue += f"DELIBERATIVE: Goal completed: {relevant_goal['name']} (reached exploration target)! "
                    chosen_action = "explore"  # Continue exploring or wait for new goal
                else:
                    agent.internal_monologue += f"DELIBERATIVE: Pursuing exploration goal '{relevant_goal['name']}'. Moving towards ({target_x},{target_y}). "
                    if dist_x > 0:
                        chosen_action = "move_down" if agent.pos_x < target_x else "move_up"
                    elif dist_y > 0:
                        chosen_action = "move_right" if agent.pos_y < target_y else "move_left"
                    else:
                        chosen_action = random.choice(
                            ["move_up", "move_down", "move_left", "move_right"])  # Random move if stuck or at target

        # 4. Semantic Memory / Working Memory Influence (More prominent in Deliberative Mode)
        # These are generally lower priority and refine the chosen action rather than override.
        if decision_mode == 'deliberative':
            # Semantic Memory influence for hunger
            if agent.internal_state.hunger > 0.6:
                food_facts = agent.semantic_memory.retrieve_facts("food")
                if food_facts:
                    agent.internal_monologue += f"DELIBERATIVE: Semantic memory reminds me that 'food' is {food_facts.get('property', 'unknown')} and {food_facts.get('effect', 'has no effect')}. "
                    # If food is in sight and we know it reduces hunger, prioritize seeking food
                    if agent.perception["food_in_sight"] and agent.semantic_memory.infer_property("food",
                                                                                                  "effect") == "reduces_hunger" and chosen_action not in [
                        "seek_food", "move_up", "move_down", "move_left", "move_right", "move_object"]:
                        agent.internal_monologue += "Food in sight and known to reduce hunger, considering 'seek_food'. "
                        chosen_action = "seek_food"

            # Semantic Memory influence for thirst
            if agent.internal_state.thirst > 0.6:
                water_facts = agent.semantic_memory.retrieve_facts("water")
                if water_facts:
                    agent.internal_monologue += f"DELIBERATIVE: Semantic memory reminds me that 'water' is {water_facts.get('property', 'unknown')} and {water_facts.get('effect', 'has no effect')}. "
                    if agent.perception["water_in_sight"] and agent.semantic_memory.infer_property("water",
                                                                                                   "effect") == "reduces_thirst" and chosen_action not in [
                        "drink_water", "move_up", "move_down", "move_left", "move_right", "move_object"]:
                        agent.internal_monologue += "Water in sight and known to reduce thirst, considering 'drink_water'. "
                        chosen_action = "drink_water"

            # Semantic Memory influence for fatigue/rest
            if agent.internal_state.fatigue > 0.6:
                rest_facts = agent.semantic_memory.retrieve_facts("rest")
                if rest_facts and rest_facts.get("effect") == "reduces_fatigue" and chosen_action != "rest":
                    agent.internal_monologue += f"DELIBERATIVE: Semantic memory reminds me that 'rest' {rest_facts.get('effect', 'has no effect')}. Prioritizing 'rest'. "
                    chosen_action = "rest"

            # Semantic Memory influence for stormy weather
            current_weather = agent.perception["current_weather"]
            if current_weather == "stormy":
                shelter_facts = agent.semantic_memory.retrieve_facts("shelter")
                if shelter_facts and shelter_facts.get("property") == "provides_safety" and shelter_facts.get(
                        "context") == "bad_weather" and chosen_action not in ["rest"]:
                    agent.internal_monologue += f"DELIBERATIVE: Semantic memory informs me that 'shelter' provides safety in bad weather. I should seek shelter (or rest). "
                    chosen_action = "rest"  # As shelter action is not yet explicit, fallback to rest

            for item in reversed(agent.working_memory_buffer):
                if item["type"] == "perceived_food" and \
                        agent.current_time_step - item["time"] <= 3 and \
                        (agent.pos_x, agent.pos_y) != item["location"]:
                    food_x, food_y = item["location"]
                    if chosen_action not in ["seek_food", "move_up", "move_down", "move_left", "move_right",
                                             "move_object"]:
                        agent.internal_monologue += f"DELIBERATIVE: Working memory recalls food at ({food_x},{food_y}). "
                        if abs(agent.pos_x - food_x) > abs(agent.pos_y - food_y):
                            chosen_action = "move_down" if agent.pos_x < food_x else "move_up"
                        else:
                            chosen_action = "move_right" if agent.pos_y < food_y else "move_left"
                        agent.internal_monologue += f"Suggesting movement towards recalled food: {chosen_action}. "
                    break
                elif item["type"] == "perceived_water" and \
                        agent.current_time_step - item["time"] <= 3 and \
                        (agent.pos_x, agent.pos_y) != item["location"]:
                    water_x, water_y = item["location"]
                    if chosen_action not in ["drink_water", "move_up", "move_down", "move_left", "move_right",
                                             "move_object"]:
                        agent.internal_monologue += f"DELIBERATIVE: Working memory recalls water at ({water_x},{water_y}). "
                        if abs(agent.pos_x - water_x) > abs(agent.pos_y - water_y):
                            chosen_action = "move_down" if agent.pos_x < water_x else "move_up"
                        else:
                            chosen_action = "move_right" if agent.pos_y < water_y else "move_left"
                        agent.internal_monologue += f"Suggesting movement towards recalled water: {chosen_action}. "
                    break

        # 5. Environmental/Emotional Modifiers (Weather, Curiosity, Other Agents)
        # These can influence both modes, but their impact might differ.
        current_weather = agent.perception["current_weather"]
        if current_weather == "stormy" and chosen_action not in ["rest"]:
            agent.internal_monologue += "Stormy weather makes me want to rest. "
            chosen_action = "rest"
        elif current_weather == "rainy" and chosen_action == "explore" and random.random() < 0.5:
            agent.internal_monologue += "Rainy weather makes me reconsider exploration. "
            chosen_action = "rest" if agent.internal_state.fatigue < 0.8 else "seek_food"

        # Curiosity-driven exploration (More prominent in Deliberative Mode if no pressing needs/goals)
        if agent.emotion_state.get("curiosity") > 0.6 and \
                agent.internal_state.hunger < 0.7 and agent.internal_state.fatigue < 0.7 and agent.internal_state.thirst < 0.7 and \
                not uncompleted_goals:
            if decision_mode == 'deliberative' and chosen_action not in ["explore", "move_up", "move_down", "move_left",
                                                                         "move_right", "move_object"]:
                agent.internal_monologue += "DELIBERATIVE: High curiosity and no pressing needs, so I will explore. "
                chosen_action = random.choice(["move_up", "move_down", "move_left", "move_right"])
            elif decision_mode == 'reactive' and chosen_action == "explore":
                agent.internal_monologue += "REACTIVE: Curiosity is active, but less emphasis on exploration in this mode. "

        # Simple social interaction (if other agents are nearby)
        if agent.perception[
            "other_agents_in_sight"] and agent.internal_state.hunger > 0.5 and chosen_action == "explore":
            agent.internal_monologue += "Seeing other agents while hungry makes me prioritize seeking food over exploring. "
            chosen_action = "seek_food"

        return chosen_action

