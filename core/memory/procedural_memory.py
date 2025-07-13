# core/memory/procedural_memory.py

from typing import Dict, Any, List, Optional


class ProceduralMemory:
    """
    Manages the agent's procedural memory, storing learned skills and habits.

    Procedural memories are often implicit (unconscious) and represent
    "how to do" things, such as sequences of actions for specific situations.
    """

    def __init__(self, capacity: int = 10):
        """
        Initializes the procedural memory.

        Args:
            capacity (int): The maximum number of procedures to store.
        """
        self.capacity: int = capacity
        # Procedures will be stored as a dictionary mapping a unique ID to the procedure details.
        # Each procedure might look like:
        # {
        #   "id": "unique_id_for_procedure",
        #   "name": "Descriptive Name",
        #   "condition": {"type": "hunger_high", "threshold": 0.8}, # Condition that triggers this procedure
        #   "action_sequence": ["seek_food", "eat"], # Sequence of actions
        #   "priority": 0.7, # How important/strong this procedure is
        #   "last_triggered_step": -1, # When it was last used
        #   "success_count": 0, # How many times it led to success
        #   "failure_count": 0, # How many times it led to failure
        #   "condition_description": "When hunger is high" # Human-readable description
        # }
        self.procedures: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 0  # Simple counter for unique procedure IDs

    def add_procedure(self, name: str, condition: Dict[str, Any], action_sequence: List[str],
                      priority: float, condition_description: str) -> str:
        """
        Adds a new procedure to the memory.

        Args:
            name (str): A descriptive name for the procedure.
            condition (Dict[str, Any]): A dictionary defining the conditions under which
                                        this procedure should be considered.
            action_sequence (List[str]): A list of actions to perform when this procedure is triggered.
            priority (float): The base priority of this procedure (0.0 to 1.0).
            condition_description (str): A human-readable description of the condition.

        Returns:
            str: The ID of the newly added procedure.
        """
        if len(self.procedures) >= self.capacity:
            # Simple eviction policy: remove the lowest priority procedure
            if self.procedures:
                lowest_priority_id = min(self.procedures, key=lambda k: self.procedures[k]["priority"])
                del self.procedures[lowest_priority_id]
                print(f"ProceduralMemory: Evicted procedure '{lowest_priority_id}' to make space.")
            else:  # Should not happen if capacity is non-zero
                print("ProceduralMemory: Cannot add procedure, memory is full and empty.")
                return ""

        new_id = f"proc_{self._next_id}"
        self._next_id += 1

        procedure = {
            "id": new_id,
            "name": name,
            "condition": condition,
            "action_sequence": action_sequence,
            "priority": priority,
            "last_triggered_step": -1,
            "success_count": 0,
            "failure_count": 0,
            "condition_description": condition_description
        }
        self.procedures[new_id] = procedure
        print(f"ProceduralMemory: Added new procedure '{name}' with ID '{new_id}'.")
        return new_id

    def get_triggered_procedure(self, agent) -> Optional[Dict[str, Any]]:
        """
        Evaluates current agent state and perceptions to find a relevant,
        high-priority procedure that should be triggered.

        Args:
            agent: The agent instance, providing access to its internal state and perception.

        Returns:
            Optional[Dict[str, Any]]: The triggered procedure dictionary, or None if no
                                      procedure's conditions are met.
        """
        best_procedure: Optional[Dict[str, Any]] = None
        highest_priority: float = -1.0

        for proc_id, procedure in self.procedures.items():
            condition_met = False
            # Evaluate the condition for each procedure
            if procedure["condition"]["type"] == "hunger_high":
                if agent.internal_state.hunger >= procedure["condition"]["threshold"]:
                    condition_met = True
            elif procedure["condition"]["type"] == "fatigue_high":
                if agent.internal_state.fatigue >= procedure["condition"]["threshold"]:
                    condition_met = True
            elif procedure["condition"]["type"] == "food_in_sight":
                if agent.perception["food_in_sight"] and agent.internal_state.hunger > 0.5:
                    condition_met = True
            elif procedure["condition"]["type"] == "obstacle_blocking_path":
                # This condition will need more sophisticated pathfinding logic.
                # For now, a simple check: if agent sees an obstacle and has a location goal
                if agent.perception["obstacle_in_sight"] and \
                        any(g["type"] == "reach_location" and not g["completed"] for g in agent.active_goals):
                    condition_met = True
            # Add more condition types as needed

            if condition_met:
                # Consider dynamic priority based on recent success/failure or emotional state
                current_priority = procedure["priority"]
                if procedure["success_count"] > procedure["failure_count"] and procedure["success_count"] > 0:
                    current_priority = min(1.0, current_priority * 1.1)  # Boost for successful procedures
                elif procedure["failure_count"] > procedure["success_count"] and procedure["failure_count"] > 0:
                    current_priority = max(0.0, current_priority * 0.8)  # Penalize for failed procedures

                if current_priority > highest_priority:
                    highest_priority = current_priority
                    best_procedure = procedure

        if best_procedure:
            # Update last triggered step for the chosen procedure
            best_procedure["last_triggered_step"] = agent.current_time_step
            # For simplicity, we return the first action in the sequence.
            # A more complex system would manage the execution of the full sequence.
            best_procedure["suggested_action"] = best_procedure["action_sequence"][0]  # Suggest the first action

        return best_procedure

    def update_procedure_outcome(self, proc_id: str, success: bool):
        """
        Updates the success/failure count for a given procedure.

        Args:
            proc_id (str): The ID of the procedure to update.
            success (bool): True if the procedure led to a successful outcome, False otherwise.
        """
        if proc_id in self.procedures:
            if success:
                self.procedures[proc_id]["success_count"] += 1
            else:
                self.procedures[proc_id]["failure_count"] += 1
            print(f"ProceduralMemory: Updated outcome for '{self.procedures[proc_id]['name']}' (Success: {success}).")

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the procedural memory content.
        """
        if not self.procedures:
            return "Procedural Memory is empty."

        s = "Procedural Memory:\n"
        for proc_id, proc in self.procedures.items():
            s += f"  - ID: {proc_id}, Name: '{proc['name']}'\n"
            s += f"    Condition: '{proc['condition_description']}'\n"
            s += f"    Action Sequence: {proc['action_sequence']}\n"
            s += f"    Priority: {proc['priority']:.2f}, Last Triggered: {proc['last_triggered_step']}\n"
            s += f"    Successes: {proc['success_count']}, Failures: {proc['failure_count']}\n"
        return s

