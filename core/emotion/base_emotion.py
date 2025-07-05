from abc import ABC, abstractmethod
from core.emotion.emotion_state import EmotionState

class EmotionStrategy(ABC):
    """
    Abstract base class (ABC) for defining different emotion calculation strategies.

    This class serves as a blueprint for how an agent's emotions should be
    updated based on its perceptions and internal states. Concrete implementations
    must inherit from this class and provide their own logic for the
    'update_emotions' method.
    """
    def __init__(self, emotion_state: EmotionState):
        """
        Initializes the EmotionStrategy with a reference to the agent's emotional state.

        Args:
            emotion_state (EmotionState): An instance of EmotionState to be managed
                                          and updated by this strategy.
        """
        self.emotion_state: EmotionState = emotion_state

    @abstractmethod
    def update_emotions(self, perception: dict, internal_state):
        """
        Abstract method to update the agent's emotions based on current perception
        and internal physiological state.

        Concrete strategy implementations must override this method to define
        their specific emotion calculation logic.

        Args:
            perception (dict): A dictionary containing the agent's current sensory
                               inputs and environmental observations (e.g., food_available, time_of_day).
            internal_state: An object representing the agent's current internal
                            physiological states (e.g., hunger, fatigue, mood).
                            Expected to be an instance of InternalState or similar.
        """
        pass