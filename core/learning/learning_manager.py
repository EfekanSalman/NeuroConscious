# core/learning/learning_manager.py

from __future__ import annotations
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from agent.base_agent import Agent
    from core.communication.communication_manager import Message


class LearningManager:
    """
    Manages the learning process for the agent.
    It handles extracting and storing new knowledge from interactions.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the learning manager with a reference to the agent.
        """
        self.agent = agent

    def process_new_information(self, message: Message):
        """
        Processes a new incoming message to extract and learn new knowledge.
        This knowledge is then stored in the agent's semantic memory.

        Args:
            message (Message): The message object from which to learn.
        """
        print(f"[{self.agent.name}]: Processing new information from message...")

        # In a real-world scenario, a sophisticated NLP model or an LLM would
        # be used here to extract key concepts or facts from the message.
        # For now, we will simulate this by simply storing the message content
        # itself as a piece of knowledge.
        new_knowledge = message.content

        # The key concept or "query" for this knowledge could be the sender's name
        # or a keyword from the message. We'll use the sender's name for simplicity.
        knowledge_query = f"Conversation with {message.sender}"

        # Store the new knowledge in the semantic memory
        self.agent.semantic_memory.store_knowledge(knowledge_query, new_knowledge)

        print(f"[{self.agent.name}]: Learned new knowledge from '{message.sender}': '{new_knowledge}'")

        # Log the learning event in the internal monologue
        self.agent.internal_monologue += f" Learned from message from {message.sender}: '{new_knowledge}'."

    def reflect(self):
        """
        Simulates the agent reflecting on its memories to form new,
        more abstract knowledge. This is a placeholder for future development.
        """
        print(f"[{self.agent.name}]: Reflecting on past experiences...")
        # TODO: Implement a mechanism to query memory, synthesize information,
        # and create new, higher-level knowledge. This would be a crucial step
        # for more advanced "thought."
        pass

