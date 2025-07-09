from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE
import json
from visualization.action_plot import plot_action_counts
from visualization.internal_state_plot import plot_internal_states
import random

# Define a filepath for saving/loading the Q-table
Q_TABLE_FILEPATH = "q_table_simbot1.json"


def main():
    # Initialize core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Initialize the World (which now has a grid)

    # Initialize Multiple Agents and add to the World
    # Agent 1
    agent1 = Agent(name="SimBot-1", mood_strategy=mood_strategy)
    world.add_agent(agent1, start_pos=(0, 0))  # Place SimBot-1 at (0,0)

    # Load Q-table for agent1 if it exists
    agent1.q_learner.load_q_table(Q_TABLE_FILEPATH)

    # Agent 2
    agent2 = Agent(name="SimBot-2", mood_strategy=mood_strategy)
    # Use the imported GRID_SIZE directly
    world.add_agent(agent2, start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))

    # Agent 3 (Optional)
    # agent3 = Agent(name="SimBot-3", mood_strategy=mood_strategy)
    # world.add_agent(agent3, start_pos=(0, GRID_SIZE - 1))

    # Initialize action counter with all possible actions from the Q-learner
    # We'll track actions for the first agent for simplicity in this plot.
    # For multi-agent plots, you might need a more complex visualization.
    action_counter = {action: 0 for action in agent1.q_learner.actions}

    # Plot initial action counts (will be all zeros)
    plot_action_counts(action_counter)

    total_steps = 1000  # Total simulation steps

    # Lists to store internal state data for plotting
    hunger_history = []
    fatigue_history = []
    mood_history = []
    time_steps = []

    print("--- Starting Multi-Agent Simulation ---")
    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through ALL agents' sense, think, act cycle.
        world.step()

        # Count the action frequency for SimBot-1 (or any agent you want to track)
        if agent1._last_performed_action:
            action_counter[agent1._last_performed_action] += 1

        # New: Collect internal state data for plotting
        hunger_history.append(agent1.internal_state.hunger)
        fatigue_history.append(agent1.internal_state.fatigue)
        mood_history.append(
            agent1.internal_state.mood_value)  # Assuming mood_value is a numerical representation of mood
        time_steps.append(step)

        # Log status for detailed output every 100 steps
        if step % 100 == 0:
            print(f"\n--- Step {step} Detailed Status ---")
            # Agent's log_status now includes its position and local grid view
            # You can log the status of all agents if desired:
            # for agent in world.agents:
            #     agent.log_status()
            agent1.log_status()  # For conciseness in the console output, we only log agent1

    print("\n--- Simulation Completed  ---")
    print("For SimBot-1", total_steps, "Action Counts after step:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Action counts saved in action_counts.json.")
    except IOError as e:
        print(f"Error while saving action numbers: {e}")

    # Save Q-table of SimBot-1 using the new method
    agent1.q_learner.save_q_table(Q_TABLE_FILEPATH)

    # Plot final action counts
    plot_action_counts(action_counter)

    # Plot internal state history
    plot_internal_states(time_steps, hunger_history, fatigue_history, mood_history, agent_name=agent1.name)


if __name__ == "__main__":
    main()

