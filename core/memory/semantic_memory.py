from typing import Dict, List, Any

class SemanticMemory:
    """
    Manages an agent's semantic memory, storing general knowledge and facts
    about the world and concepts.

    Unlike episodic memory (which stores specific events), semantic memory
    stores generalized knowledge that can influence reasoning and decision-making.
    """
    def __init__(self):
        """
        Initializes the semantic memory.
        Knowledge is stored as a dictionary, where keys are concepts/topics
        and values are lists of associated facts or attributes.
        """
        self.knowledge_base: Dict[str, List[str]] = {}

        # Initialize with some basic, innate knowledge
        self._initialize_innate_knowledge()

    def _initialize_innate_knowledge(self):
        """
        Adds some pre-defined, innate knowledge to the agent's semantic memory.
        This represents knowledge the agent is born with or has acquired through
        early, non-simulated learning.
        """
        self.add_fact("food", "Food reduces hunger.")
        self.add_fact("food", "Food is found in the environment.")
        self.add_fact("rest", "Rest reduces fatigue.")
        self.add_fact("rest", "Rest is important for recovery.")
        self.add_fact("exploration", "Exploration can lead to new discoveries.")
        self.add_fact("exploration", "Exploration might increase hunger and fatigue.")
        self.add_fact("weather", "Stormy weather is unpleasant and might hinder movement.")
        self.add_fact("weather", "Rainy weather can be slightly unpleasant.")
        self.add_fact("goals", "Achieving goals is rewarding.")
        self.add_fact("other_agents", "Other agents exist in the world.")
        self.add_fact("other_agents", "Other agents might compete for resources.")
        self.add_fact("self", "I need to maintain low hunger and fatigue.")
        self.add_fact("self", "I feel emotions like joy, fear, frustration, and curiosity.")


    def add_fact(self, concept: str, fact: str):
        """
        Adds a new fact to the semantic memory under a given concept.

        Args:
            concept (str): The main concept or topic this fact relates to (e.g., "food", "rest").
            fact (str): The factual statement to store.
        """
        if concept not in self.knowledge_base:
            self.knowledge_base[concept] = []
        if fact not in self.knowledge_base[concept]: # Avoid duplicate facts
            self.knowledge_base[concept].append(fact)
            # print(f"Semantic Memory: Added fact '{fact}' under '{concept}'.")

    def retrieve_facts(self, concept: str) -> List[str]:
        """
        Retrieves all facts associated with a given concept.

        Args:
            concept (str): The concept to query.

        Returns:
            List[str]: A list of facts related to the concept. Returns an empty list if no facts found.
        """
        return self.knowledge_base.get(concept, [])

    def __str__(self) -> str:
        """
        Provides a string representation of the semantic memory content.
        """
        s = "Semantic Memory:\n"
        if not self.knowledge_base:
            s += "  (Empty)"
        else:
            for concept, facts in self.knowledge_base.items():
                s += f"  {concept.capitalize()}:\n"
                for fact in facts:
                    s += f"    - {fact}\n"
        return s

