from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World, GRID_SIZE
import json
from visualization.action_plot import plot_action_counts
import random


def main():
    # Initialize core components
    mood_strategy = BasicMoodStrategy()
    world = World()  # Start the world (it now has a grid)

    # Launch and Add Multiple Agents to the World
    # Agent 1
    agent1 = Agent(name="SimBot-1", mood_strategy=mood_strategy)
    world.add_agent(agent1, start_pos=(0, 0))

    # Agent 2
    agent2 = Agent(name="SimBot-2", mood_strategy=mood_strategy)
    # Place in opposite corner using GRID_SIZE directly
    world.add_agent(agent2, start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))

    # Agent 3
    agent3 = Agent(name="SimBot-3", mood_strategy=mood_strategy)
    world.add_agent(agent3, start_pos=(0, GRID_SIZE - 1))

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

        # Log status for detailed output every 100 steps
        if step % 100 == 0:
            print(f"\n--- Adım {step} Detaylı Durum ---")
            # Agent's log_status now includes its location and local grid view
            # You can log the status of all agents if desired:
            # for agent in world.agents:
            #     agent.log_status()
            agent1.log_status()  # For brevity in the console output, we only log agent1

    print("\n--- Simülasyon Tamamlandı ---")
    print("SimBot-1 için", total_steps, "adımdan sonraki Aksiyon Sayıları:")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

    # Save action counts to JSON
    try:
        with open("action_counts.json", "w") as f:
            json.dump(action_counter, f, indent=2)
        print("Aksiyon sayıları action_counts.json dosyasına kaydedildi.")
    except IOError as e:
        print(f"Aksiyon sayıları kaydedilirken hata oluştu: {e}")

    # Dump SimBot-1's Q-table to JSON
    try:
        # Convert 'defaultdict' to a regular dict for JSON serialization
        # Since q_table contains nested dictionaries, we also need to transform the values
        q_table_serializable = {
            state_key: dict(actions_dict)
            for state_key, actions_dict in agent1.q_learner.q_table.items()
        }
        with open("q_table_simbot1.json", "w") as f:  # Renamed for specific agent
            json.dump(q_table_serializable, f, indent=2)
        print("SimBot-1'in Q-tablosu q_table_simbot1.json dosyasına kaydedildi.")
    except IOError as e:
        print(f"Q-tablosu kaydedilirken hata oluştu: {e}")
    except TypeError as e:
        print(
            f"Q-tablosu JSON'a serileştirilirken hata oluştu: {e}. Tüm anahtarların/değerlerin JSON-serileştirilebilir olduğundan emin olun.")

    # Draw the last action numbers
    plot_action_counts(action_counter)


if __name__ == "__main__":
    main()
