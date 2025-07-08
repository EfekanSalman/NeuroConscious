from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE  # Import GRID_SIZE directly
import json
from visualization.action_plot import plot_action_counts
import random


def main():
    # Initialize core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Initialize the World (which now has a grid)

    # Initialize Multiple Agents and add to the World
    # Agent 1
    agent1 = Agent(name="SimBot-1", mood_strategy=mood_strategy)
    world.add_agent(agent1, start_pos=(0, 0))  # Place SimBot-1 at (0,0)

    # Agent 2
    agent2 = Agent(name="SimBot-2", mood_strategy=mood_strategy)
    # Use the imported GRID_SIZE directly
    world.add_agent(agent2, start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))  # Place SimBot-2 at opposite corner

    # Agent 3
    agent3 = Agent(name="SimBot-3", mood_strategy=mood_strategy)
    world.add_agent(agent3, start_pos=(0, GRID_SIZE - 1)) # Place SimBot-3 at another corner

    # Initialize action counter with all possible actions from the Q-learner
    action_counter = {action: 0 for action in agent1.q_learner.actions}

    # Plot initial action counts (will be all zeros)
    plot_action_counts(action_counter)

    total_steps = 1000  # Total simulation steps

    print("--- Starting Multi-Agent Simulation ---")
    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through ALL agents' sense, think, act cycle.
        world.step()

        # Count action frequency for SimBot-1
        if agent1._last_performed_action:
            action_counter[agent1._last_performed_action] += 1

        # Log status every 100 steps for detailed output
        if step % 100 == 0:
            print(f"\n--- Detailed Status at Step {step} ---")
            # Agent's log_status now includes its position and local grid view
            # for agent in world.agents:
            #     agent.log_status()
            agent1.log_status()  # Logging only agent1 for brevity in console output

    print("\n--- Simulation Complete ---")
    print("Action Counts for SimBot-1 after", total_steps, "steps:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Action counts saved to action_counts.json")
    except IOError as e:
        print(f"Error saving action counts: {e}")

    # Dump Q-table of SimBot-1 to JSON
    try:
        q_table_serializable = {
            state_key: dict(actions_dict)
            for state_key, actions_dict in agent1.q_learner.q_table.items()
        }
        with open("q_table_simbot1.json", "w") as f:  # Renamed for specific agent
            json.dump(q_table_serializable, f, indent=2)
        print("Q-table for SimBot-1 saved to q_table_simbot1.json")
    except IOError as e:
        print(f"Error saving Q-table: {e}")
    except TypeError as e:
        print(f"Error serializing Q-table to JSON: {e}. Ensure all keys/values are JSON-serializable.")

    # Plot final action counts
    plot_action_counts(action_counter)


if __name__ == "__main__":
    main()
