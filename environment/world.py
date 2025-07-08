import random
from typing import List
from agent.base_agent import Agent

# --- World Configuration Constants ---
GRID_SIZE = 10 # The world will be a 10x10 grid
FOOD_SPAWN_CHANCE = 0.1 # Chance for food to appear in an empty cell per update
MAX_FOOD_ITEMS = 5 # Maximum number of food items that can exist on the grid at once

# Weather related constants
WEATHER_TYPES = ["sunny", "cloudy", "rainy", "stormy"]
WEATHER_CHANGE_CHANCE = 0.2 # 20% chance for weather to change at each step
WEATHER_CYCLE_LENGTH = 20 # How many steps before a weather type tends to cycle (optional, for more structured changes)

class World:
    """
    Represents the simulated environment in which agents exist and interact.

    The World class manages a grid-based map, global time step, environmental conditions
    (like food availability, time of day, and weather), and orchestrates the actions of all
    agents within it. It serves as the central hub for the NeuroConscious simulation.
    """
    def __init__(self):
        """
        Initializes the World environment with a grid map and initial weather.

        Sets up an empty list for agents, initializes the time step, and
        establishes initial environmental conditions including the grid map and weather.
        """
        self.agents: List[Agent] = []       # A list to hold all agents participating in this world.
        self.time_step: int = 0             # The current simulation time step.
        self.food_available: bool = False   # Flag indicating if food is currently available anywhere on the grid.
        self.time_of_day: str = "day"       # Current time of day, either "day" or "night".
        self.current_weather: str = random.choice(WEATHER_TYPES) # Initial random weather

        # Initialize the grid map
        # 'grid' will be a 2D list representing the world.
        # Each cell can contain 'empty', 'food', 'obstacle', etc.
        self.grid: List[List[str]] = [['empty' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self._place_initial_food() # Place some food at the start

    def _place_initial_food(self):
        """
        Places a few initial food items randomly on the grid.
        """
        food_placed = 0
        while food_placed < MAX_FOOD_ITEMS / 2: # Place half of max food initially
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.grid[row][col] == 'empty':
                self.grid[row][col] = 'food'
                food_placed += 1

    def add_agent(self, agent: Agent, start_pos: tuple[int, int] = None):
        """
        Adds an agent to the world and sets this world as the agent's environment.
        Places the agent on the grid.

        Args:
            agent (Agent): An instance of the Agent class to be added to the simulation.
            start_pos (tuple[int, int], optional): The (row, col) coordinates to place the agent.
                                                   If None, a random empty spot will be chosen.
        """
        # Ensure agent has position attributes
        if not hasattr(agent, 'pos_x') or not hasattr(agent, 'pos_y'):
            print(f"Warning: Agent {agent.name} does not have pos_x or pos_y attributes. Adding them.")
            agent.pos_x = 0
            agent.pos_y = 0

        if start_pos:
            row, col = start_pos
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.grid[row][col] == 'empty':
                agent.pos_x = row
                agent.pos_y = col
                # self.grid[row][col] = 'agent' # Optional: Mark agent's position on grid
            else:
                print(f"Warning: Invalid or occupied start position {start_pos} for agent {agent.name}. Placing randomly.")
                self._place_agent_randomly(agent)
        else:
            self._place_agent_randomly(agent)

        agent.set_environment(self) # Link the agent back to this world instance.
        self.agents.append(agent)   # Add the agent to the world's list of active agents.

    def _place_agent_randomly(self, agent: Agent):
        """Helper to place an agent at a random empty spot on the grid."""
        while True:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            # For simplicity, agents can share a cell with 'empty' or 'food'
            if self.grid[row][col] in ['empty', 'food']:
                agent.pos_x = row
                agent.pos_y = col
                # self.grid[row][col] = 'agent' # Optional: Mark agent's position on grid
                break

    def update_environment(self):
        """
        Updates the environmental conditions for the current time step,
        including food regeneration, day/night cycle, and weather changes.
        """
        # Regenerate food in empty cells with a certain chance, up to MAX_FOOD_ITEMS
        current_food_count = sum(row.count('food') for row in self.grid)
        if current_food_count < MAX_FOOD_ITEMS:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if self.grid[r][c] == 'empty' and random.random() < FOOD_SPAWN_CHANCE:
                        self.grid[r][c] = 'food'
                        current_food_count += 1
                        if current_food_count >= MAX_FOOD_ITEMS:
                            break
                if current_food_count >= MAX_FOOD_ITEMS:
                    break

        # Update the global food_available flag based on actual food on the grid
        self.food_available = current_food_count > 0

        # Simulate a simple day/night cycle.
        # It toggles every 5 steps, creating 5 steps of day and 5 steps of night within a 10-step cycle.
        if self.time_step % 10 < 5:
            self.time_of_day = "day"
        else:
            self.time_of_day = "night"

        # Simulate weather changes
        # There's a WEATHER_CHANGE_CHANCE at each step for the weather to change randomly.
        if random.random() < WEATHER_CHANGE_CHANCE:
            self.current_weather = random.choice(WEATHER_TYPES)

    def step(self):
        """
        Executes a single simulation step.

        During each step, the environment is updated, and then each agent
        in the world perceives its surroundings, thinks (decides an action),
        and acts upon that decision. Agent status is logged before actions.
        """
        self.update_environment() # Update global conditions like food, time of day, and weather.
        # Log the current time step and environmental conditions.
        print(f"\n=== Time Step {self.time_step} | Food: {self.food_available} | Time: {self.time_of_day} | Weather: {self.current_weather} ===")
        self.print_grid()

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

    def print_grid(self):
        """
        Prints a visual representation of the grid, including agent positions.
        'F' for food, 'A' for agent, '.' for empty.
        If an agent is on food, it will show 'A'.
        """
        display_grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Populate food and other static elements
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] == 'food':
                    display_grid[r][c] = 'F'
                # Add other elements like obstacles them later
                # elif self.grid[r][c] == 'obstacle':
                #     display_grid[r][c] = '#'

        # Place agents on the display grid
        for agent in self.agents:
            # Check if agent has position attributes before trying to access them
            if hasattr(agent, 'pos_x') and hasattr(agent, 'pos_y'):
                row, col = agent.pos_x, agent.pos_y
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    # Mark agent's position. Agent overrides other elements for display.
                    display_grid[row][col] = 'A'
            else:
                print(f"Warning: Agent {agent.name} has no position to display.")


        print("\n--- Current Grid ---")
        for row in display_grid:
            print(" ".join(row))
        print("--------------------")

