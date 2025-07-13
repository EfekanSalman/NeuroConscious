# core/memory/semantic_memory.py
from random import random
from typing import Dict, Any, List, Optional


class SemanticMemory:
    """
    Manages the agent's semantic memory, storing general knowledge and facts
    about the world. This memory is typically explicit and represents "what"
    things are, their properties, and relationships.
    """

    def __init__(self, capacity: int = 50):
        """
        Initializes the semantic memory.

        Args:
            capacity (int): The maximum number of facts/relations to store.
        """
        self.capacity: int = capacity
        # Facts will be stored as a dictionary, where keys could be entities or concepts,
        # and values are dictionaries of properties or relationships.
        # Example:
        # {
        #   "food": {"is_a": "resource", "property": "edible", "effect": "reduces_hunger"},
        #   "water": {"is_a": "resource", "property": "drinkable", "effect": "reduces_thirst"},
        #   "obstacle": {"is_a": "barrier", "property": "immovable_by_default", "action_needed": "move_object"},
        #   "apple": {"is_a": "food", "color": "red", "location_type": "tree"},
        #   "tree": {"has_property": "provides_shade", "contains": "apple"}
        # }
        self.facts: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 0  # Simple counter for unique fact IDs (if needed for internal tracking)

    def add_fact(self, entity: str, properties: Dict[str, Any]):
        """
        Adds or updates a fact about an entity in the semantic memory.

        Args:
            entity (str): The name of the entity or concept (e.g., "food", "apple").
            properties (Dict[str, Any]): A dictionary of properties or relationships
                                         associated with the entity.
        """
        if entity not in self.facts and len(self.facts) >= self.capacity:
            # Simple eviction policy: remove a random old fact if capacity is reached
            if self.facts:
                entity_to_evict = random.choice(list(self.facts.keys()))
                del self.facts[entity_to_evict]
                print(f"SemanticMemory: Evicted fact about '{entity_to_evict}' to make space.")
            else:
                print("SemanticMemory: Cannot add fact, memory is full and empty.")
                return

        if entity in self.facts:
            # Update existing properties
            self.facts[entity].update(properties)
            print(f"SemanticMemory: Updated fact about '{entity}'.")
        else:
            self.facts[entity] = properties
            print(f"SemanticMemory: Added new fact about '{entity}'.")

    def retrieve_facts(self, query_entity: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all known facts about a specific entity.

        Args:
            query_entity (str): The entity or concept to query (e.g., "food").

        Returns:
            Optional[Dict[str, Any]]: A dictionary of properties/relationships for the
                                      queried entity, or None if not found.
        """
        return self.facts.get(query_entity, None)

    def infer_property(self, entity: str, property_name: str) -> Any:
        """
        Attempts to infer a property about an entity, either directly or via relationships.
        This is a very basic inference mechanism.

        Args:
            entity (str): The entity to infer about.
            property_name (str): The name of the property to infer (e.g., "is_a", "effect").

        Returns:
            Any: The inferred property value, or None if it cannot be inferred.
        """
        # 1. Direct lookup
        if entity in self.facts and property_name in self.facts[entity]:
            return self.facts[entity][property_name]

        # 2. Simple 'is_a' inheritance (e.g., if 'apple' is_a 'food', and 'food' has 'edible' property)
        if "is_a" in self.facts.get(entity, {}):
            parent_entity = self.facts[entity]["is_a"]
            if parent_entity in self.facts and property_name in self.facts[parent_entity]:
                return self.facts[parent_entity][property_name]

        # Add more complex inference rules here as needed (e.g., transitive relations)

        return None  # Property not found or cannot be inferred

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the semantic memory content.
        """
        if not self.facts:
            return "Semantic Memory is empty."

        s = "Semantic Memory:\n"
        for entity, properties in self.facts.items():
            s += f"  - {entity.capitalize()}:\n"
            for prop, value in properties.items():
                s += f"    - {prop}: {value}\n"
        return s

