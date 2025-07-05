import random
from collections import defaultdict

class MotivationEngine:
    """
    A foundational or alternative motivation engine for an agent,
    primarily focusing on basic physiological needs and simple memory.

    This engine determines an agent's actions based on hunger, fatigue,
    and whether food has been recently perceived, offering a direct
    response mechanism to immediate internal states.
    """
    def __init__(self, state):
        """
        Initializes the MotivationEngine with a reference to the agent's internal state.

        Args:
            state: An object representing the agent's current internal physiological
                   states (e.g., an InternalState instance with 'hunger' and 'fatigue' attributes).
        """
        # Reference to the agent's internal state system.
        self.internal_state = state

    def decide_action(self, perception: dict = None, memory: dict = None) -> str:
        """
        Decides the agent's next action based on its current physiological needs
        and simple environmental perceptions or memories.

        This method prioritizes actions to satisfy high hunger or fatigue,
        with a basic consideration of food availability from perception or recent memory.

        Args:
            perception (dict, optional): A dictionary containing the agent's current
                                         sensory inputs (e.g., "food_available").
                                         Defaults to None, but typically expected.
            memory (dict, optional): A dictionary containing simple memory items
                                     (e.g., "food_last_seen", "current_step").
                                     Defaults to None, but typically expected.

        Returns:
            str: The chosen action (e.g., "seek_food", "rest", "explore").
        """
        # Ensure perception and memory are dictionaries to prevent AttributeError if None
        perception = perception if perception is not None else {}
        memory = memory if memory is not None else {}

        # Retrieve current hunger and fatigue levels from the internal state.
        current_hunger = self.internal_state.hunger
        current_fatigue = self.internal_state.fatigue

        # Default action if no specific needs are pressing.
        action = "explore"

        # --- Memory-based and Perception-based Decision Logic ---
        # Check current food availability from perception.
        food_available_now = perception.get("food_available", False)
        # Check when food was last seen from memory.
        food_last_seen_step = memory.get("food_last_seen", None)
        # Get the current simulation step from memory.
        current_simulation_step = memory.get("current_step", 0)

        # Determine if food was seen recently (within the last 3 time steps).
        food_recently_seen = (
            food_last_seen_step is not None and
            (current_simulation_step - food_last_seen_step) <= 3
        )

        # Prioritize actions based on internal needs:
        # If hunger is high, decide between seeking food or exploring.
        if current_hunger > 0.6:
            if food_available_now or food_recently_seen:
                # If food is available now or was seen recently, prioritize seeking it.
                action = "seek_food"
            else:
                # If hungry but no immediate food cue, explore to find it.
                action = "explore"
        # If fatigue is high (and hunger is not a primary concern, or less so).
        elif current_fatigue > 0.6:
            # Prioritize resting to reduce fatigue.
            action = "rest"

        return action