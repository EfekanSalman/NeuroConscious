from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World
import json
from visualization.action_plot import plot_action_counts

def main():
    mood_strategy = BasicMoodStrategy()
    agent = Agent(name="SimBot", mood_strategy=mood_strategy)
    world = World()
    world.add_agent(agent)


    action_counter = {
        "seek_food": 0,
        "rest": 0,
        "explore": 0
    }
    plot_action_counts(action_counter)
    total_steps = 1000

    for step in range(total_steps):
        world.update_environment()
        agent.sense()
        action = agent.think()
        agent.act(action)

        # Count action frequency
        action_counter[action] += 1

        if step % 100 == 0:
            print(f"\n--- Step {step} ---")
            agent.log_status()

    print("\n Simulation complete.")
    print("Action Counts after 1000 steps:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    with open("action_counts.json", "w") as f:
        json.dump(action_counter, f, indent=2)

    # Dumped version
    with open("q_table.json", "w") as f:
        json.dump(agent.q_learner.q_table, f, indent=2)


if __name__ == "__main__":
    main()
