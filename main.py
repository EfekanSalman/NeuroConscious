from agent.base_agent import Agent
from core.mood.basic_mood import BasicMoodStrategy
from environment.world import World


def run_simulation(steps = 10):
    mood_strategy = BasicMoodStrategy()
    agent = Agent(name = "SimBot", mood_strategy = mood_strategy)

    world = World()
    world.add_agent(agent)

    world.run(steps = 15)

if __name__ == "__main__":
    run_simulation()
