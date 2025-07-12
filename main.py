from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE
import json
from visualization.action_plot import plot_action_counts
from visualization.internal_state_plot import plot_internal_states
from visualization.emotion_plot import plot_emotion_history  # Import for plotting emotion history
import random

# Define a filepath for saving/loading the DQN model
DQN_MODEL_FILEPATH = "dqn_model_simbot1.pth"  # .pth is a common extension for PyTorch models


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
    mood_history = []
    time_steps = []

    # Updated: Dictionary to store emotion history for plotting with lowercase keys
    emotion_history = {
        "joy": [],  # Changed to lowercase
        "fear": [],  # Changed to lowercase
        "frustration": [],  # Fixed: Removed extra double quote
        "curiosity": []  # Changed to lowercase
    }

    print("--- Starting Multi-Agent Simulation ---")
    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through ALL agents' sense, think, act cycle.
        world.step()

        # Count action frequency for SimBot-1 (or any agent you want to track)
        if agent1._last_performed_action:
            action_counter[agent1._last_performed_action] += 1

        # Collect internal state data for plotting
        hunger_history.append(agent1.internal_state.hunger)
        fatigue_history.append(agent1.internal_state.fatigue)
        mood_history.append(agent1.internal_state.mood_value)
        time_steps.append(step)

        # Collect emotion data for plotting
        emotion_state = agent1.emotion_state.get_emotions()  # Get current emotion values as a dict
        for emotion_name, value in emotion_state.items():
            # The check `if emotion_name in emotion_history` will now correctly match lowercase keys
            if emotion_name in emotion_history:
                emotion_history[emotion_name].append(value)

        # Log detailed status every 100 steps
        if step % 100 == 0:
            print(f"\n--- Step {step} Detailed Status ---")
            # Agent's log_status now includes its position and local grid view
            # You can log all agents' status if desired:
            # for agent in world.agents:
            #     agent.log_status()
            agent1.log_status()  # Logging only agent1 for brevity in console output

    print("\n--- Simulation Completed ---")
    print("Action Counts after", total_steps, "steps for SimBot-1:")
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
    plot_internal_states(time_steps, hunger_history, fatigue_history, mood_history, agent_name=agent1.name)

    # Plot emotion history
    plot_emotion_history(time_steps, emotion_history, agent_name=agent1.name)


if __name__ == "__main__":
    main()

