# main.py

from core.mood.basic_mood import BasicMoodStrategy
from agent.base_agent import Agent
from environment.world import World
from visualization.plotter import AgentLogger

def main():
    mood_strategy = BasicMoodStrategy()
    agent = Agent(name="SimBot", mood_strategy=mood_strategy)
    logger = AgentLogger(agent_name=agent.name)

    world = World()
    world.add_agent(agent)

    for step in range(20):
        world.update_environment()
        agent.sense()
        action = agent.think()
        agent.act(action)

        logger.log(step, agent.state.hunger, agent.state.fatigue, agent.state.mood)

    logger.plot()

if __name__ == "__main__":
    main()
