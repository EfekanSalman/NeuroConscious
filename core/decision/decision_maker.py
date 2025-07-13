import random
from typing import Dict, Any, Tuple


class DecisionMaker:
    """
    Manages the agent's high-level decision-making process, acting as a central
    arbiter for action selection.

    It takes inputs from various cognitive modules (DQN, needs, goals, memories)
    and applies a hierarchical logic or set of rules to determine the final action.
    This version incorporates deliberative and reactive decision-making modes.
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

        # --- Decision Hierarchy based on Mode ---

        # Reactive Mode prioritizes immediate needs and strong procedural memories.
        # Deliberative Mode prioritizes goals, planning, and learned values (DQN).

        # 1. Critical Needs Override (Highest Priority in both modes, but more immediate in Reactive)
        if agent.internal_state.hunger > 0.85:
            chosen_action = "seek_food"
            agent.internal_monologue += "CRITICAL: Extreme hunger detected, prioritizing 'seek_food'. "
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
            fatigue=agent.internal_state.fatigue
        )
        chosen_action = initial_dqn_suggestion
        agent.internal_monologue += f"Defaulting to DQN suggestion: {chosen_action}. "

        # 3. Goal System Influence (More prominent in Deliberative Mode)
        uncompleted_goals = [g for g in agent.active_goals if not g["completed"]]
        if uncompleted_goals and decision_mode == 'deliberative':
            highest_priority_goal = max(uncompleted_goals, key=lambda g: g["priority"])

            if highest_priority_goal["type"] == "reach_location":
                dist_x = abs(agent.pos_x - highest_priority_goal["target_x"])
                dist_y = abs(agent.pos_y - highest_priority_goal["target_y"])
                distance = dist_x + dist_y

                if distance == 0:
                    highest_priority_goal["completed"] = True
                    agent.internal_monologue += f"Goal completed: {highest_priority_goal['name']}! "
                    # Reward for goal completion can be applied here or in act()
                elif distance <= 5:  # Deliberative mode can plan for slightly further goals
                    agent.internal_monologue += f"DELIBERATIVE: Goal '{highest_priority_goal['name']}' is within reach ({distance} units), prioritizing movement. "
                    if dist_x > 0:
                        chosen_action = "move_down" if agent.pos_x < highest_priority_goal["target_x"] else "move_up"
                    elif dist_y > 0:
                        chosen_action = "move_right" if agent.pos_y < highest_priority_goal["target_y"] else "move_left"

            elif highest_priority_goal["type"] == "maintain_hunger_low":
                if agent.internal_state.hunger < highest_priority_goal["threshold"]:
                    highest_priority_goal["current_duration"] += 1
                    if highest_priority_goal["current_duration"] >= highest_priority_goal["duration_steps"]:
                        highest_priority_goal["completed"] = True
                        agent.internal_monologue += f"DELIBERATIVE: Goal completed: {highest_priority_goal['name']} (maintained hunger)! "
                    if agent.internal_state.hunger >= highest_priority_goal[
                        "threshold"] * 0.8 and chosen_action != "seek_food":
                        agent.internal_monologue += f"DELIBERATIVE: Hunger approaching threshold for goal '{highest_priority_goal['name']}', suggesting 'seek_food'. "
                        chosen_action = "seek_food"
                else:
                    highest_priority_goal["current_duration"] = 0

        # 4. Semantic Memory / Working Memory Influence (More prominent in Deliberative Mode)
        # These are generally lower priority and refine the chosen action rather than override.
        if decision_mode == 'deliberative':
            if agent.internal_state.hunger > 0.6:
                food_facts = agent.semantic_memory.retrieve_facts("food")
                if food_facts:
                    agent.internal_monologue += "DELIBERATIVE: Semantic memory reminds me about food. "
                    if agent.perception["food_in_sight"] and chosen_action not in ["seek_food", "move_up", "move_down",
                                                                                   "move_left", "move_right"]:
                        agent.internal_monologue += "Food in sight, considering 'seek_food'. "

                for item in reversed(agent.working_memory_buffer):
                    if item["type"] == "perceived_food" and \
                            agent.current_time_step - item["time"] <= 3 and \
                            (agent.pos_x, agent.pos_y) != item["location"]:
                        food_x, food_y = item["location"]
                        if chosen_action not in ["seek_food", "move_up", "move_down", "move_left", "move_right"]:
                            agent.internal_monologue += f"DELIBERATIVE: Working memory recalls food at ({food_x},{food_y}). "
                            if abs(agent.pos_x - food_x) > abs(agent.pos_y - food_y):
                                chosen_action = "move_down" if agent.pos_x < food_x else "move_up"
                            else:
                                chosen_action = "move_right" if agent.pos_y < food_y else "move_left"
                            agent.internal_monologue += f"Suggesting movement towards recalled food: {chosen_action}. "
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
                agent.internal_state.hunger < 0.7 and agent.internal_state.fatigue < 0.7 and \
                not uncompleted_goals:  # and chosen_action not in movement/explore
            if decision_mode == 'deliberative' and chosen_action not in ["explore", "move_up", "move_down", "move_left",
                                                                         "move_right"]:
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

