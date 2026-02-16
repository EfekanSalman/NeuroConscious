# core/motivation/basic_motivation.py

import random

class BasicMotivationEngine:
    def __init__(self, agent):
        self.agent = agent  # Access internal state if needed

    def decide_action(self, perception: dict, memory, emotions):
        """
        Decide the best action based on internal needs and emotional influence.
        Emotions: joy, fear, curiosity, frustration (range: 0.0 - 1.0)
        """
        hunger = self.agent.state.hunger
        fatigue = self.agent.state.fatigue

        # Get emotion values
        joy = emotions.get("joy")
        fear = emotions.get("fear")
        frustration = emotions.get("frustration")
        curiosity = emotions.get("curiosity")

        # Debug print (optional)
        print(f"[MOTIVATION] Hunger: {hunger:.2f}, Fatigue: {fatigue:.2f}")
        print(f"[MOTIVATION] Emotions - Joy: {joy:.2f}, Fear: {fear:.2f}, Frustration: {frustration:.2f}, Curiosity: {curiosity:.2f}")

        # 1. Emergency override: high frustration forces need satisfaction
        if frustration > 0.7:
            if hunger > fatigue:
                return "seek_food"
            else:
                return "rest"

        # 2. Fear inhibits exploration; encourages safety (rest or wait)
        if fear > 0.6:
            if fatigue > 0.4:
                return "rest"
            return "seek_food" if hunger > 0.4 else "rest"

        # 3. Curiosity promotes exploration if basic needs are low
        if curiosity > 0.6 and hunger < 0.3 and fatigue < 0.3:
            return "explore"

        # 4. Joy randomly encourages non-urgent exploration
        if joy > 0.7 and random.random() < 0.3:
            return "explore"

        # 5. Default fallback based on physical needs
        if hunger > 0.6:
            return "seek_food"
        elif fatigue > 0.6:
            return "rest"

        # 6. If no strong drive, explore randomly
        return "explore"
