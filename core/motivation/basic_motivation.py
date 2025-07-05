import random

class BasicMotivationEngine:
    """
    Implements a basic motivation engine that drives agent behavior
    based on internal physiological needs and emotional states.

    This engine prioritizes actions by considering emergency overrides (high frustration),
    emotional influences (fear, curiosity, joy), and fundamental physical needs
    (hunger, fatigue).
    """
    def __init__(self, agent):
        """
        Initializes the BasicMotivationEngine with a reference to the agent.

        Args:
            agent: The Agent instance this motivation engine belongs to.
                   It's used to access the agent's internal state (hunger, fatigue).
        """
        self.agent = agent  # Reference to the main agent object.

    def decide_action(self, perception: dict, memory, emotions) -> str:
        """
        Decides the agent's next action by prioritizing various motivational factors.

        The decision-making process follows a hierarchy:
        1. Emergency actions based on high frustration.
        2. Safety-driven actions influenced by fear.
        3. Exploration driven by curiosity when basic needs are met.
        4. Random exploration influenced by high joy.
        5. Direct satisfaction of primary physiological needs (hunger, fatigue).
        6. Default exploration if no strong drives are present.

        Args:
            perception (dict): The agent's current sensory inputs and environmental observations.
            memory: (Currently unused in this implementation, but kept for potential future use).
                    Represents the agent's memory (e.g., EpisodicMemory, ShortTermMemory).
            emotions: An object containing the agent's current emotional levels
                      (e.g., EmotionState instance with 'joy', 'fear', 'curiosity', 'frustration' attributes).

        Returns:
            str: The chosen action (e.g., "seek_food", "rest", "explore").
        """
        # Retrieve current physiological states from the agent.
        current_hunger = self.agent.internal_state.hunger
        current_fatigue = self.agent.internal_state.fatigue

        # Retrieve current emotional values.
        # Emotions are expected to be between 0.0 and 1.0.
        joy = emotions.get("joy")
        fear = emotions.get("fear")
        frustration = emotions.get("frustration")
        curiosity = emotions.get("curiosity")

        # --- Debugging Output (Optional) ---
        # These print statements provide real-time insights into the decision-making factors.
        print(f"[MOTIVATION] Hunger: {current_hunger:.2f}, Fatigue: {current_fatigue:.2f}")
        print(f"[MOTIVATION] Emotions - Joy: {joy:.2f}, Fear: {fear:.2f}, Frustration: {frustration:.2f}, Curiosity: {curiosity:.2f}")
        # -----------------------------------

        # 1. Emergency Override: High frustration forces immediate need satisfaction.
        #    If the agent is highly frustrated, it will prioritize reducing the higher need.
        if frustration > 0.7:
            print("[MOTIVATION] High frustration detected, prioritizing immediate needs.")
            if current_hunger > current_fatigue:
                return "seek_food"
            else:
                return "rest"

        # 2. Fear Inhibition: High fear discourages risky actions like exploration; encourages safety.
        #    If fearful, the agent opts for resting (if tired) or seeking food (if hungry)
        #    to mitigate perceived threats by stabilizing internal state.
        if fear > 0.6:
            print("[MOTIVATION] High fear detected, prioritizing safety.")
            if current_fatigue > 0.4: # Prioritize rest if sufficiently tired under fear
                return "rest"
            return "seek_food" if current_hunger > 0.4 else "rest" # Otherwise, seek food or rest as secondary safety

        # 3. Curiosity-Driven Exploration: Promotes exploration when basic needs are low.
        #    If the agent is not hungry or tired, curiosity can lead to exploring the environment.
        if curiosity > 0.6 and current_hunger < 0.3 and current_fatigue < 0.3:
            print("[MOTIVATION] Curiosity is high, basic needs low, exploring.")
            return "explore"

        # 4. Joy-Influenced Exploration: High joy can lead to random, non-urgent exploration.
        #    A happy agent might occasionally explore just for the sake of it.
        if joy > 0.7 and random.random() < 0.3:
            print("[MOTIVATION] High joy detected, randomly choosing to explore.")
            return "explore"

        # 5. Default Fallback: Prioritize basic physical needs if they are high.
        #    If no emotional overrides or specific curiosities, address primary needs.
        if current_hunger > 0.6:
            print("[MOTIVATION] Hunger is high, seeking food.")
            return "seek_food"
        elif current_fatigue > 0.6:
            print("[MOTIVATION] Fatigue is high, resting.")
            return "rest"

        # 6. Default Action: If no strong drive, explore.
        #    As a last resort, if no other conditions are met, the agent will explore.
        print("[MOTIVATION] No strong specific drive, defaulting to exploration.")
        return "explore"