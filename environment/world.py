import random
from typing import List
from agent.base_agent import Agent

class World:
    def __init__(self):
        self.agents: List[Agent] = []
        self.time_step = 0
        self.food_available = False
        self.time_of_day = "day" #day or night

    def add_agent(self, agent: Agent):
        agent.set_environment(self)
        self.agents.append(agent)

    def update_environment(self):
        self.food_available = random.random() < 0.5 # 50% chance

        # Simulate day/night cycle (every 5 steps toggle)
        if self.time_step % 10 < 5:
            self.time_of_day = "day"
        else:
            self.time_of_day = "night"

    def step(self):
        self.update_environment()
        print(f"\n=== Time Step {self.time_step} | Food: {self.food_available} | Time: {self.time_of_day} ===")

        for agent in self.agents:
            agent.log_status()
            agent.sense()
            action = agent.think()
            print(f"{agent.name} decided to {action}")
            agent.act(action)

        self.time_step += 1

    def run(self, steps: int = 10):
        # The underscore (_) is used here as a throwaway variable.
        # It indicates that the loop variable is intentionally unused.
        # Simply want to repeat the step() method 'steps' times.
        for _ in range(steps):
            self.step()
