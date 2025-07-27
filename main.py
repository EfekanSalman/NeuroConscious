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
# LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
import os  # For directory cleanup if needed, but not for temp_frames anymore

# Define a filepath for saving/loading the DQN model
DQN_MODEL_FILEPATH = "dqn_model_simbot1.pth"


def main():
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
    thirst_history = []  # New: List to store thirst history
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

    # Removed: temp_frames_dir creation and cleanup
    print("--- Simulation Starting ---")

    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through ALL agents' sense, think, act cycle.
        # Removed: save_frame=True parameter from world.step()
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
        thirst_history.append(agent1.internal_state.thirst)  # New: Collect thirst history
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
    # New: Pass thirst_history to plot_internal_states
    plot_internal_states(time_steps, hunger_history, fatigue_history, thirst_history, mood_history,
                         agent_name=agent1.name)

    # Plot emotion history
    plot_emotion_history(time_steps, emotion_history, agent_name=agent1.name)

    # Plot action timeline
    plot_action_timeline(time_steps[:len(action_history)], action_history, consciousness_state_history,
                         agent_name=agent1.name)

    # Plot DQN Q-values
    # New: We will assume a fixed thirst level (e.g., 0.5) for this 2D visualization
    plot_dqn_q_values(agent1.q_learner, agent_name=agent1.name, fixed_thirst_level=0.5)

if __name__ == "__main__":
    main()

