from core.mood.base import MoodStrategy

class InternalState:
    """
    Represents the internal physiological and psychological state of an agent.

    This includes core needs like hunger and fatigue, and a derived emotional
    or psychological state like mood. Values are typically normalized between 0.0 and 1.0.
    """
    def __init__(self, mood_strategy: MoodStrategy, hunger: float = 0.5, fatigue: float = 0.5):
        """
        Initializes the agent's internal state.

        Args:
            mood_strategy (MoodStrategy): The strategy used to calculate the agent's mood.
            hunger (float, optional): Initial hunger level (0.0 = full, 1.0 = starving). Defaults to 0.5.
            fatigue (float, optional): Initial fatigue level (0.0 = rested, 1.0 = exhausted). Defaults to 0.5.
        """
        self._hunger: float = self._clamp_value(hunger)
        self._fatigue: float = self._clamp_value(fatigue)
        self._mood_strategy: MoodStrategy = mood_strategy
        self._mood: str = "neutral" # Initial mood as string
        self._mood_value: float = 0.5 # New: Numerical representation of mood (e.g., 0.0 to 1.0)
        self._update_mood() # Calculate initial mood and mood_value

    def _clamp_value(self, value: float) -> float:
        """Clamps a value between 0.0 and 1.0."""
        return max(0.0, min(1.0, value))

    @property
    def hunger(self) -> float:
        """Gets the current hunger level."""
        return self._hunger

    @hunger.setter
    def hunger(self, value: float):
        """Sets the hunger level and clamps it between 0.0 and 1.0."""
        self._hunger = self._clamp_value(value)
        self._update_mood() # Recalculate mood when hunger changes

    @property
    def fatigue(self) -> float:
        """Gets the current fatigue level."""
        return self._fatigue

    @fatigue.setter
    def fatigue(self, value: float):
        """Sets the fatigue level and clamps it between 0.0 and 1.0."""
        self._fatigue = self._clamp_value(value)
        self._update_mood() # Recalculate mood when fatigue changes

    @property
    def mood(self) -> str:
        """Gets the current mood (string representation)."""
        return self._mood

    @property
    def mood_value(self) -> float:
        """Gets the current numerical mood value (0.0 to 1.0)."""
        return self._mood_value

    def _update_mood(self):
        """
        Calculates and updates the agent's mood (both string and numerical)
        based on current hunger and fatigue levels using the assigned mood strategy.
        """
        self._mood = self._mood_strategy.calculate_mood(self._hunger, self._fatigue)
        # Map string mood to a numerical value for plotting/further calculations
        if self._mood == "happy":
            self._mood_value = 1.0
        elif self._mood == "neutral":
            self._mood_value = 0.5
        elif self._mood == "sad":
            self._mood_value = 0.0
        elif self._mood == "frustrated": # Assuming frustrated is a negative mood
            self._mood_value = 0.2
        elif self._mood == "anxious": # Assuming anxious is a negative mood
            self._mood_value = 0.3
        else: # Default for any other unmapped mood
            self._mood_value = 0.5


    def update(self, delta_time: float):
        """
        Updates the agent's internal state over time.

        Hunger and fatigue naturally increase over time, simulating metabolic processes.
        Mood is recalculated after these updates.

        Args:
            delta_time (float): The time elapsed since the last update.
        """
        # Hunger and fatigue increase over time
        self.hunger = self._hunger + (0.01 * delta_time) # Hunger increases by 0.01 per delta_time
        self.fatigue = self._fatigue + (0.005 * delta_time) # Fatigue increases by 0.005 per delta_time
        # Mood is automatically updated by the setters for hunger/fatigue,
        # but calling it explicitly here ensures it's always fresh after a general update.
        self._update_mood()

    def snapshot(self):
        """
        Creates a snapshot (copy) of the current internal state.

        This is useful for storing the state before an action is taken,
        for use in learning algorithms (e.g., Q-learning previous state).

        Returns:
            InternalState: A new InternalState object with the current values.
        """
        # Create a new instance of InternalState with current values
        # Pass the same mood_strategy instance to the snapshot
        return InternalState(self._mood_strategy, hunger=self._hunger, fatigue=self._fatigue)

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the internal state.
        """
        return f"Hunger: {self._hunger:.2f}, Fatigue: {self._fatigue:.2f}, Mood: {self._mood}"

