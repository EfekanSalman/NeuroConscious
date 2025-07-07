# main.py

from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World
import json
from visualization.action_plot import plot_action_counts
import random  # Import random for initial agent placement


def main():
    # Initialize core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Initialize the World (which now has a grid)

    # Initialize Agent and add to the World
    agent = Agent(name="SimBot", mood_strategy=mood_strategy)
    world.add_agent(agent)

    # Initialize action counter with all possible actions from the Q-learner
    action_counter = {action: 0 for action in agent.q_learner.actions}

    # Plot initial action counts (will be all zeros)
    plot_action_counts(action_counter)

    total_steps = 1000  # Total simulation steps

    print("--- Starting Simulation ---")
    for step in range(total_steps):
        # The world's step method now handles updating environment, printing grid,
        # and iterating through agents' sense, think, act cycle.
        world.step()

        # Count action frequency from the last action performed by the agent
        # Ensure _last_performed_action is set by the agent's think method
        if agent._last_performed_action:
            action_counter[agent._last_performed_action] += 1

        # Log status every 100 steps for detailed output
        if step % 100 == 0:
            print(f"\n--- Detailed Status at Step {step} ---")
            # Agent's log_status now includes its position and local grid view
            agent.log_status()

    print("\n--- Simulation Complete ---")
    print("Action Counts after", total_steps, "steps:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Action counts saved to action_counts.json")
    except IOError as e:
        print(f"Error saving action counts: {e}")

    # Dump Q-table to JSON
    try:
        # Convert defaultdict to a regular dict for JSON serialization
        # The q_table contains nested dictionaries, so we need to convert values too
        q_table_serializable = {
            state_key: dict(actions_dict)
            for state_key, actions_dict in agent.q_learner.q_table.items()
        }
        with open("q_table.json", "w") as f:
            json.dump(q_table_serializable, f, indent=2)
        print("Q-table saved to q_table.json")
    except IOError as e:
        print(f"Error saving Q-table: {e}")
    except TypeError as e:
        print(f"Error serializing Q-table to JSON: {e}. Ensure all keys/values are JSON-serializable.")

    # Plot final action counts
    plot_action_counts(action_counter)


if __name__ == "__main__":
    main()

