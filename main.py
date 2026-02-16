"""
Main entry point — runs the NeuroConscious simulation.

Uses World.step() as the single source of truth for the simulation loop.
"""

import json
import os
from collections import defaultdict

from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World
from visualization.action_plot import plot_action_counts
from config.constants import (
    TOTAL_STEPS,
    LOG_INTERVAL,
    OUTPUT_DIR,
    ACTION_COUNTS_FILE,
    Q_TABLE_FILE,
)


def main() -> None:
    """Run the simulation for TOTAL_STEPS and persist results."""
    # Setup
    mood_strategy = BasicMoodStrategy()
    agent = Agent(name="SimBot", mood_strategy=mood_strategy)
    world = World()
    world.add_agent(agent)

    action_counter: dict[str, int] = defaultdict(int)

    # Simulation loop — single orchestration point
    for step in range(TOTAL_STEPS):
        world.step()

        # Track the action that was taken this tick
        if agent._last_action:
            action_counter[agent._last_action] += 1

        if step % LOG_INTERVAL == 0:
            print(f"\n--- Step {step} ---")
            agent.log_status()

    # Results
    print("\n✅ Simulation complete.")
    print(f"🔢 Action Counts after {TOTAL_STEPS} steps:")
    for action, count in sorted(action_counter.items()):
        print(f"  {action}: {count}")

    # Persist outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, ACTION_COUNTS_FILE), "w") as f:
        json.dump(dict(action_counter), f, indent=2)

    with open(os.path.join(OUTPUT_DIR, Q_TABLE_FILE), "w") as f:
        json.dump(dict(agent.q_learner.q_table), f, indent=2)

    # Plot after simulation completes (not before)
    plot_action_counts(action_counter)


if __name__ == "__main__":
    main()
