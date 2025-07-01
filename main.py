from agent.base_agent import Agent
from core.state import InternalState
from core.mood.basic_mood import BasicMoodStrategy
from core.motivation import MotivationEngine

def run_simulation(steps = 10):
    mood_strategy = BasicMoodStrategy()
    agent = Agent(name = "SimBot", mood_strategy = mood_strategy)

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        agent.log_status()

        action = agent.think()
        print(f"Action: {action}")

        agent.act(action)

if __name__ == "__main__":
    run_simulation()
