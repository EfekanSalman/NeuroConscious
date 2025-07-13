# core/memory/procedural_memory.py

from typing import Dict, List, Any, Callable


class ProceduralMemory:
    """
    Manages an agent's procedural memory, storing learned skills, habits,
    or automated action sequences.

    Procedural memory allows the agent to execute actions or sequences of actions
    efficiently in response to specific conditions, without explicit deliberation.
    """

    def __init__(self):
        """
        Initializes the procedural memory.
        Procedures are stored as a dictionary mapping a 'trigger' condition
        to a 'suggested_action'.
        A procedure is a dict: {"condition_description": str, "condition_func": Callable, "suggested_action": str, "priority": float}
        """
        self.procedures: List[Dict[str, Any]] = []

        # Initialize with some basic, innate or pre-learned procedures
        self._initialize_innate_procedures()

    def _initialize_innate_procedures(self):
        """
        Adds some pre-defined, innate or commonly learned procedures.
        These are simplified examples; in a real system, these would be learned.
        """
        # Procedure 1: If very hungry and food is in sight, seek food.
        self.add_procedure(
            condition_description="If very hungry and food is in sight",
            condition_func=lambda agent: agent.internal_state.hunger > 0.7 and agent.perception["food_in_sight"],
            suggested_action="seek_food",
            priority=0.9  # High priority
        )
        # Procedure 2: If very fatigued, rest.
        self.add_procedure(
            condition_description="If very fatigued",
            condition_func=lambda agent: agent.internal_state.fatigue > 0.8,
            suggested_action="rest",
            priority=0.85  # High priority
        )
        # Procedure 3: If a location goal is active and close, move towards it.
        self.add_procedure(
            condition_description="If location goal is active and close",
            condition_func=self._is_location_goal_active_and_close,
            suggested_action="move_towards_goal",  # Special action to be handled by agent
            priority=0.95  # Very high priority
        )
        # Procedure 4: If feeling curious and not critical needs, explore.
        self.add_procedure(
            condition_description="If curious and needs are low",
            condition_func=lambda agent: agent.emotion_state.get("curiosity") > 0.7 and \
                                         agent.internal_state.hunger < 0.5 and \
                                         agent.internal_state.fatigue < 0.5,
            suggested_action="explore",
            priority=0.5
        )

    def _is_location_goal_active_and_close(self, agent) -> bool:
        """Helper function for a procedure condition: checks if a location goal is active and close."""
        for goal in agent.active_goals:
            if not goal["completed"] and goal["type"] == "reach_location":
                dist_x = abs(agent.pos_x - goal["target_x"])
                dist_y = abs(agent.pos_y - goal["target_y"])
                distance = dist_x + dist_y
                if distance <= 2:  # "Close" means within 2 units
                    return True
        return False

    def add_procedure(self, condition_description: str, condition_func: Callable, suggested_action: str,
                      priority: float):
        """
        Adds a new procedural rule to the memory.

        Args:
            condition_description (str): A human-readable description of the condition.
            condition_func (Callable): A function that takes the agent as an argument
                                       and returns True if the condition is met.
            suggested_action (str): The action or type of action to suggest when the condition is met.
            priority (float): The priority of this procedure (higher means more likely to override).
        """
        self.procedures.append({
            "condition_description": condition_description,
            "condition_func": condition_func,
            "suggested_action": suggested_action,
            "priority": priority
        })

    def get_triggered_procedure(self, agent) -> Dict[str, Any] | None:
        """
        Evaluates all procedures and returns the one with the highest priority
        whose condition is met.

        Args:
            agent: The agent instance whose state will be evaluated.

        Returns:
            Dict[str, Any] | None: The triggered procedure dictionary, or None if no procedure is triggered.
        """
        triggered_procedures = []
        for procedure in self.procedures:
            try:
                if procedure["condition_func"](agent):
                    triggered_procedures.append(procedure)
            except Exception as e:
                print(f"Error evaluating procedure condition '{procedure['condition_description']}': {e}")
                continue

        if not triggered_procedures:
            return None

        # Return the procedure with the highest priority
        return max(triggered_procedures, key=lambda p: p["priority"])

    def __str__(self) -> str:
        """
        Provides a string representation of the procedural memory content.
        """
        s = "Procedural Memory:\n"
        if not self.procedures:
            s += "  (Empty)"
        else:
            for proc in self.procedures:
                s += f"  - Condition: '{proc['condition_description']}' -> Action: '{proc['suggested_action']}' (Priority: {proc['priority']:.2f})\n"
        return s

