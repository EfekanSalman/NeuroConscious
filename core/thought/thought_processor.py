#!/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from agent.base_agent import Agent
    from core.consciousness.consciousness_state import ConsciousnessState
    from core.communication.communication_manager import Message


class ThoughtProcessor:
    """
    The main processor for the agent's thoughts. It manages consciousness states
    and provides the primary thinking interface for the agent.
    This version integrates with the learning manager and mood strategy.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the thought processor.
        """
        self.agent = agent

    def get_default_state(self) -> ConsciousnessState:
        """
        Returns the default consciousness state for the agent.
        """
        from core.consciousness.consciousness_state import ConsciousnessState
        return ConsciousnessState(self.agent)

    def handle_incoming_message(self, message: Message):
        """
        Processes an incoming message. It first updates the agent's mood,
        then learns from the message, retrieves relevant information from
        semantic memory, and finally generates a cognitive response.

        Args:
            message (Message): The message object received by the agent.
        """
        print(f"[{self.agent.name}]: Processing new message...")

        # Step 1: Update the agent's mood based on the message content.
        self.agent.mood_strategy.update_mood(message)

        # Step 2: Process the new message to learn from it.
        # The learning manager will store the new knowledge in semantic memory.
        self.agent.learning_manager.process_new_information(message)

        # Step 3: Now, retrieve relevant information from semantic memory,
        # which now includes the new knowledge we just stored.
        current_mood = self.agent.mood_strategy.get_mood_state()
        query = message.content
        retrieved_memory = self.agent.semantic_memory.retrieve_knowledge(query)

        # Step 4: Generate a response influenced by mood and memory.
        response_template = "Received message from {sender}: '{message_content}'."
        response_template += " My current mood is '{mood}'."

        if retrieved_memory:
            response_template += " I also remember something about this: '{memory_info}'."
            response_content = response_template.format(
                sender=message.sender,
                message_content=message.content,
                mood=current_mood,
                memory_info=retrieved_memory
            )
        else:
            response_template += " No related memories found."
            response_content = response_template.format(
                sender=message.sender,
                message_content=message.content,
                mood=current_mood
            )

        self.agent.internal_monologue += f" {response_content}"
        print(f"[{self.agent.name}]: Response generated using mood '{current_mood}' and memory.")
        print(f"[{self.agent.name}]: Response: '{response_content}'")

        # TODO: Implement a call to an LLM API here to generate a more dynamic response.
        # This is a placeholder for future development.

