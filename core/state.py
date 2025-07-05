from config.constants import DEFAULT_HUNGER_INCREASE, DEFAULT_FATIGUE_INCREASE
from core.mood.base import MoodStrategy

class InternalState:
    """
    Manages the internal physiological and emotional state of an agent.

    This class tracks core needs such as hunger and fatigue, and updates the
    agent's overall mood based on these internal conditions using a provided
    mood calculation strategy.
    """
    def __init__(self, mood_strategy: MoodStrategy, hunger: float = 0.5, fatigue: float = 0.5):
        """
        Initializes the agent's internal state.

        Args:
            mood_strategy (MoodStrategy): The strategy object used to calculate the agent's mood.
            hunger (float, optional): The initial hunger level. Clamped between 0.0 and 1.0. Defaults to 0.5.
            fatigue (float, optional): The initial fatigue level. Clamped between 0.0 and 1.0. Defaults to 0.5.
        """
        # Hunger level (0.0 = not hungry, 1.0 = starving)
        self.hunger: float = hunger
        # Fatigue level (0.0 = rested, 1.0 = exhausted)
        self.fatigue: float = fatigue
        # Current emotional mood, derived from physiological states.
        self.mood: str = "neutral"
        # Strategy object responsible for calculating the mood.
        self.mood_strategy: MoodStrategy = mood_strategy

        # Ensure initial hunger and fatigue are within valid bounds (0.0 to 1.0)
        self.hunger = min(1.0, max(0.0, self.hunger))
        self.fatigue = min(1.0, max(0.0, self.fatigue))
        # Initialize mood based on the initial hunger/fatigue
        self.mood = self.mood_strategy.calculate_mood(self.hunger, self.fatigue)


    def update(self, delta_time: float = 1.0):
        """
        Updates the agent's hunger and fatigue levels over time, then recalculates its mood.

        Hunger and fatigue naturally increase as time passes. The mood is then
        updated based on these new physiological levels using the assigned
        mood strategy.

        Args:
            delta_time (float, optional): The time increment for this update cycle.
                                          Affects how much hunger and fatigue increase. Defaults to 1.0.
        """
        # Hunger increases over time, capped at 1.0 (maximum).
        self.hunger = min(1.0, self.hunger + DEFAULT_HUNGER_INCREASE * delta_time)
        # Fatigue increases over time, capped at 1.0 (maximum).
        self.fatigue = min(1.0, self.fatigue + DEFAULT_FATIGUE_INCREASE * delta_time)

        # Recalculate mood based on the updated hunger and fatigue levels.
        self.mood = self.mood_strategy.calculate_mood(self.hunger, self.fatigue)

    def snapshot(self):
        """
        Creates a new InternalState object that is a copy of the current state.

        This is useful for storing historical states (e.g., for learning algorithms)
        without modifying the original object. The new snapshot will share the same
        mood_strategy object reference.

        Returns:
            InternalState: A new InternalState object with the same hunger, fatigue,
                           and mood values as the current instance.
        """
        # Create a new instance of the same type, passing the mood_strategy reference.
        # Then, copy the physiological and mood values from the current instance.
        return type(self)(
            self.mood_strategy
        )._copy_values_from(self)

    def _copy_values_from(self, other) -> 'InternalState':
        """
        Internal helper method to copy hunger, fatigue, and mood values from another
        InternalState instance to this instance.

        Args:
            other (InternalState): The InternalState object from which to copy values.

        Returns:
            InternalState: The current InternalState instance, with updated values.
        """
        self.hunger = other.hunger
        self.fatigue = other.fatigue
        self.mood = other.mood
        return self

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the agent's internal state.

        Useful for logging and debugging, showing current hunger, fatigue, and mood.

        Returns:
            str: A formatted string showing the current hunger, fatigue, and mood levels.
                 Example: "Hunger: 0.75, Fatigue: 0.30, Mood: neutral"
        """
        return f"Hunger: {self.hunger:.2f}, Fatigue: {self.fatigue:.2f}, Mood: {self.mood}"