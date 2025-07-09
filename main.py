from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE
import json
from visualization.action_plot import plot_action_counts
import random

# Define a filepath for saving/loading the Q-table
Q_TABLE_FILEPATH = "q_table_simbot1.json"


def main():
    # Launch core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Initialize the world (now with a grid)

    # Launch and Add Multiple Agents to the World
    # Agent 1
    agent1 = Agent(name="SimBot-1", mood_strategy=mood_strategy)
    world.add_agent(agent1, start_pos=(0, 0))  # Place SimBot-1 at position (0,0)

    # Load Q-table for agent1 if it exists
    agent1.q_learner.load_q_table(Q_TABLE_FILEPATH)

    # Agent 2
    agent2 = Agent(name="SimBot-2", mood_strategy=mood_strategy)
    # Place in opposite corner using GRID_SIZE directly
    world.add_agent(agent2, start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))

    # Agent 3 (Optional)
    # agent3 = Agent(name="SimBot-3", mood_strategy=mood_strategy)
    # world.add_agent(agent3, start_pos=(0, GRID_SIZE - 1))

    # Start the action counter with all possible actions from the Q-learner
    action_counter = {action: 0 for action in agent1.q_learner.actions}

    # Draw the starting action numbers (all zeros)
    plot_action_counts(action_counter)

    total_steps = 1000  # Total number of simulation steps

    print("--- Multi-Agent Simulation Launching ---")
    for step in range(total_steps):
        # The world's step method now includes updating the perimeter, printing the grid
        # and manages the sensing, thinking, acting cycle of ALL agents.
        world.step()

        # Count the action frequency for SimBot-1 (or any agent you want to track)
        if agent1._last_performed_action:
            action_counter[agent1._last_performed_action] += 1

        #  Log status for detailed output every 100 steps
        if step % 100 == 0:
            print(f"\n--- Step {step} Detailed Status ---")
            # Agent's log_status now includes its location and local grid view
            # You can log the status of all agents if desired:
            # for agent in world.agents:
            #     agent.log_status()
            agent1.log_status()  # For brevity in the console output, we only log agent1

    print("\n--- Simulation Completed ---")
    print("For SimBot-1“, total_steps, ”Number of Actions after step:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Action counts saved in action_counts.json.")
    except IOError as e:
        print(f"Error saving action numbers: {e}")

    # Save Q-table of SimBot-1 using the new method
    agent1.q_learner.save_q_table(Q_TABLE_FILEPATH)

    # Plot final action counts
    plot_action_counts(action_counter)


if __name__ == "__main__":
    main()

