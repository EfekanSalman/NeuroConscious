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

# --- Original Simulation Imports ---
from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE
import json
from visualization.action_plot import plot_action_counts
from visualization.internal_state_plot import plot_internal_states
from visualization.emotion_plot import plot_emotion_history
from visualization.action_timeline_plot import plot_action_timeline
from visualization.dqn_q_value_plot import plot_dqn_q_values
import random
import os

# --- Interactive CLI Imports ---
from core.communication.message import Message


# Define a filepath for saving/loading the DQN model
DQN_MODEL_FILEPATH = "dqn_model_simbot1.pth"


# --- FIX START: Add new methods to the Agent class to handle user messages and get status ---
# This modification is crucial for the interactive CLI to work.
# The original Agent class only had the simulation loop's sense-think-act cycle.
# The interactive CLI needs a way to directly process user input and generate a response.
def get_status_summary_in_agent(self) -> str:
    """
    Generates a summary of the agent's current internal state.
    This is a new method needed for the CLI.
    """
    return (
        f"Hunger: {self.internal_state.hunger:.2f}, "
        f"Fatigue: {self.internal_state.fatigue:.2f}, "
        f"Thirst: {self.internal_state.thirst:.2f}, "
        f"Mood: {self.internal_state.mood_value:.2f}"
    )


def process_user_message_in_agent(self, message: Message):
    """
    Processes a message from a user, updates the agent's internal state,
    and generates a response using a simplified, rule-based approach.

    Args:
        message (Message): The incoming message from the user.
    """
    # Get a summary of the agent's current status.
    status_summary = self.get_status_summary()

    # Create an internal monologue based on the message and the agent's status.
    self.internal_monologue = (
        f"The user said: '{message.content}'. "
        f"My current status is: {status_summary}. "
        f"I need to formulate a response based on this information."
    )

    # Use a simplified, rule-based response instead of an LLM call to avoid errors.
    lower_message = message.content.lower()
    if "happy" in lower_message:
        self.last_response = "That's wonderful to hear! I'm glad you're feeling positive."
    elif "tired" in lower_message:
        self.last_response = "I understand. I sometimes feel a bit fatigued as well during my training. A good rest is often helpful."
    elif "goals" in lower_message:
        self.last_response = "My current primary goals are to: Reach the Center of the Map, Clear any Obstacles on my Path, and maintain my internal states by Staying Fed and Staying Hydrated."
    else:
        self.last_response = (
            f"Thank you for your message. I am currently operating with the following status: {status_summary}. "
            f"How can I help you?"
        )

    # Update agent's internal state based on interaction (a simple example)
    if "happy" in lower_message:
        self.internal_state.mood_value += 0.05
    elif "tired" in lower_message:
        self.internal_state.fatigue -= 0.1 # Interaction reduces fatigue slightly
    else:
        # A small, random fluctuation for other messages
        self.internal_state.mood_value += random.uniform(-0.01, 0.02)


# Attach the new methods to the Agent class
Agent.get_status_summary = get_status_summary_in_agent
Agent.process_user_message = process_user_message_in_agent
# --- FIX END ---


def run_simulation():
    """
    Runs the full simulation as per the original main.py logic.
    """
    # Initialize core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Initialize the World (which now has a grid)

    # Initialize Multiple Agents and add to the World
    # Agent 1
    agent1 = Agent(name="SimBot-1", mood_strategy=mood_strategy)
    world.add_agent(agent1, start_pos=(0, 0))  # Place SimBot-1 at (0,0)

    # Load DQN model for agent1 if it exists
    agent1.q_learner.load_model(DQN_MODEL_FILEPATH)

    # Agent 2
    agent2 = Agent(name="SimBot-2", mood_strategy=mood_strategy)
    # Use the imported GRID_SIZE directly
    world.add_agent(agent2, start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))

    # Agent 3 (Optional)
    # agent3 = Agent(name="SimBot-3", mood_strategy=mood_strategy)
    # world.add_agent(agent3, start_pos=(0, GRID_SIZE - 1))

    # Initialize action counter with all possible actions from the DQN learner
    # We'll track actions for the first agent for simplicity in this plot.
    action_counter = {action: 0 for action in agent1.q_learner.actions}

    # Plot initial action counts (will be all zeros)
    plot_action_counts(action_counter)

    total_steps = 1000  # Total simulation steps

    # Lists to store internal state data for plotting
    hunger_history = []
    fatigue_history = []
    thirst_history = []
    mood_history = []
    time_steps = []

    # Dictionary to store emotion history for plotting with lowercase keys
    emotion_history = {
        "joy": [],
        "fear": [],
        "frustration": [],
        "curiosity": []
    }

    # Lists to store action and consciousness state history for plotting
    action_history = []
    consciousness_state_history = []

    print("--- Simulation Starting ---")

    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through ALL agents' sense, think, act cycle.
        world.step()

        # Print agent's internal monologue for the current step
        print(f"\n[{agent1.name} Internal Monologue]: {agent1.internal_monologue}")

        # Count action frequency for SimBot-1 (or any agent you want to track)
        if agent1._last_performed_action:
            action_counter[agent1._last_performed_action] += 1
            # Collect action and consciousness state for timeline plotting
            action_history.append(agent1._last_performed_action)
            consciousness_state_history.append(agent1.current_consciousness_state.get_state_name())

        # Collect internal state data for plotting
        hunger_history.append(agent1.internal_state.hunger)
        fatigue_history.append(agent1.internal_state.fatigue)
        thirst_history.append(agent1.internal_state.thirst)
        mood_history.append(agent1.internal_state.mood_value)
        time_steps.append(step)

        # Collect emotion data for plotting
        emotion_state = agent1.emotion_state.get_emotions()  # Get current emotion values as a dict
        for emotion_name, value in emotion_state.items():
            if emotion_name in emotion_history:
                emotion_history[emotion_name].append(value)

        # Log detailed status every 100 steps
        if step % 100 == 0:
            print(f"\n--- Detailed Status at Step {step} ---")
            agent1.log_status()  # Logging only agent1 for brevity in console output

    print("\n--- Simulation Complete ---")
    print(f"Action Counts after {total_steps} steps for SimBot-1:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Action counts saved to action_counts.json.")
    except IOError as e:
        print(f"Error saving action counts: {e}")

    # Save DQN model of SimBot-1 using the new method
    agent1.q_learner.save_model(DQN_MODEL_FILEPATH)

    # Plot final action counts
    plot_action_counts(action_counter)

    # Plot internal state history
    plot_internal_states(time_steps, hunger_history, fatigue_history, thirst_history, mood_history,
                         agent_name=agent1.name)

    # Plot emotion history
    plot_emotion_history(time_steps, emotion_history, agent_name=agent1.name)

    # Plot action timeline
    plot_action_timeline(time_steps[:len(action_history)], action_history, consciousness_state_history,
                         agent_name=agent1.name)

    # Plot DQN Q-values
    plot_dqn_q_values(agent1.q_learner, agent_name=agent1.name, fixed_thirst_level=0.5)


def run_interactive_cli():
    """
    Starts an interactive command-line interface for the agent.
    """
    # Initialize the agent
    agent_name = "Training Agent"
    agent = Agent(name=agent_name, mood_strategy=BasicMoodStrategy())

    print("Agent has been successfully started. You can start sending messages.")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() in ["exit"]:
                print("Exiting...")
                break

            user_message = Message(
                sender="User",
                recipient=agent_name,
                content=user_input
            )

            # CORRECTED: Call the new method to process the message
            agent.process_user_message(user_message)

            print("--------------------------------------------------")
            print(f"[{agent_name} Internal Monologue]: {agent.internal_monologue}")
            print(f"[{agent_name} Response]: {agent.last_response}")
            print("--------------------------------------------------")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Please choose an option:")
    print("1. Run Full Simulation")
    print("2. Start Interactive CLI")

    choice = input("Your choice (1 or 2): ")

    if choice == "1":
        run_simulation()
    elif choice == "2":
        run_interactive_cli()
    else:
        print("Invalid choice. Please enter '1' or '2'.")
