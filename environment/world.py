import random
from typing import List
from agent.base_agent import Agent # Assuming base_agent.py contains the Agent class

class World:
    """
    Represents the simulated environment in which agents exist and interact.

    The World class manages the global time step, environmental conditions (like food availability
    and time of day), and orchestrates the actions of all agents within it. It serves
    as the central hub for the NeuroConscious simulation.
    """
    def __init__(self):
        """
        Initializes the World environment.

        Sets up an empty list for agents, initializes the time step, and
        establishes initial environmental conditions.
        """
        self.agents: List[Agent] = []       # A list to hold all agents participating in this world.
        self.time_step: int = 0             # The current simulation time step.
        self.food_available: bool = False   # Flag indicating if food is currently available in the environment.
        self.time_of_day: str = "day"       # Current time of day, either "day" or "night".

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the world and sets this world as the agent's environment.

        Args:
            agent (Agent): An instance of the Agent class to be added to the simulation.
        """
        agent.set_environment(self) # Link the agent back to this world instance.
        self.agents.append(agent)   # Add the agent to the world's list of active agents.

    def update_environment(self):
        """
        Updates the environmental conditions for the current time step.

        This includes randomly determining food availability and simulating
        a day/night cycle based on the current time step.
        """
        # Randomly set food availability: 50% chance food is available at each step.
        self.food_available = random.random() < 0.5

        # Simulate a simple day/night cycle.
        # It toggles every 5 steps, creating 5 steps of day and 5 steps of night within a 10-step cycle.
        if self.time_step % 10 < 5:
            self.time_of_day = "day"
        else:
            self.time_of_day = "night"

    def step(self):
        """
        Executes a single simulation step.

        During each step, the environment is updated, and then each agent
        in the world perceives its surroundings, thinks (decides an action),
        and acts upon that decision. Agent status is logged before actions.
        """
        self.update_environment() # Update global conditions like food and time of day.
        # Log the current time step and environmental conditions.
        print(f"\n=== Time Step {self.time_step} | Food: {self.food_available} | Time: {self.time_of_day} ===")

        for agent in self.agents:
            agent.log_status()   # Display agent's current internal state.
            agent.sense()        # Agent perceives the updated environment.
            action = agent.think() # Agent processes perceptions and decides an action.
            print(f"{agent.name} decided to {action}") # Log the agent's chosen action.
            agent.act(action)    # Agent performs the chosen action, affecting its internal state.

        self.time_step += 1 # Advance the global simulation time step.

    def run(self, steps: int = 10):
        """
        Runs the simulation for a specified number of steps.

        Args:
            steps (int, optional): The total number of simulation steps to run. Defaults to 10.
        """
        # Iterate 'steps' times, calling the 'step()' method for each iteration.
        # The underscore (_) is used as a convention for a loop variable that is not used inside the loop.
        for _ in range(steps):
            self.step()