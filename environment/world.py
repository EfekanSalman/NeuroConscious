# !/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# environment/world.py

import random
from typing import List, Optional, Tuple, Any
from agent.base_agent import Agent
from core.navigation.regions import half_of as _half_of

# --- World Configuration Constants ---
GRID_SIZE = 10  # The world will be a 10x10 grid
FOOD_SPAWN_CHANCE = 0.1  # Chance for food to appear in an empty cell per update
MAX_FOOD_ITEMS = 5  # Maximum number of food items that can exist on the grid at once

WATER_SPAWN_CHANCE = 0.05  # Chance for water to appear
MAX_WATER_ITEMS = 5  # Maximum number of water items (equal to MAX_FOOD_ITEMS: water is as plentiful as food)

# Order in which empty cells get a chance to receive regenerated food/water: 'uniform' (random order, so
# resources reappear anywhere) or 'row_major' (legacy: top rows first, which piles resources into the top rows).
RESPAWN_ORDER = "uniform"

# Fixed resource sites (0 = off, the default: resources then regenerate anywhere). With N sites a kind of
# resource only ever grows on N cells chosen once per world, and a consumed one regrows in the same cell
# after SITE_REGROW_STEPS, so where food and water are stays worth remembering.
FOOD_SITES = 0
WATER_SITES = 0
SITE_REGROW_STEPS = 50
# Beacon regime (0 = off, the default): with fixed sites, only the sites in one half of the grid (columns
# below GRID_SIZE / 2 = 'left', the rest 'right') hold resources; every BEACON_CYCLE steps the active half is
# redrawn at random and the sites of the other half are emptied. Which half is active is the cue.
BEACON_CYCLE = 0
# Irregular redraws: each cycle lasts BEACON_CYCLE * uniform(1 - jitter, 1 + jitter) steps (0 = every BEACON_CYCLE steps).
BEACON_CYCLE_JITTER = 0.0

OBSTACLE_SPAWN_CHANCE = 0.01  # Chance for obstacles to appear
MAX_OBSTACLES = 2  # Maximum number of obstacles

WEATHER_TYPES = ["sunny", "rainy", "stormy"]  # Possible weather conditions


def half_of(col: int) -> str:
    return _half_of(col, GRID_SIZE)


class World:
    """
    Represents the simulated environment in which agents exist and interact.

    The World class manages a grid-based map, global time step, environmental conditions
    (like food availability, water availability, time of day, and weather), and orchestrates
    the actions of all agents within it. It serves as the central hub for the NeuroConscious simulation.
    This version includes water resources, static obstacles, dynamic weather, and object movement.
    """

    def __init__(self):
        """
        Initializes the World environment with a grid map.

        Sets up an empty list for agents, initializes the time step, and
        establishes initial environmental conditions including the grid map.
        """
        self.agents: List[Agent] = []  # A list to hold all agents participating in this world.
        self.time_step: int = 0  # The current simulation time step.
        self.food_available: bool = False  # Flag indicating if food is currently available anywhere on the grid.
        self.water_available: bool = False  # Flag indicating if water is available
        self.time_of_day: str = "day"  # Current time of day, either "day" or "night".
        self.current_weather: str = "sunny"  # Current weather condition

        # Initialize the grid map
        # 'grid' will be a 2D list representing the world.
        # Each cell can contain 'empty', 'food', 'water', 'obstacle', etc.
        self.grid: List[List[str]] = [['empty' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self._sites = {'food': [], 'water': []}  # Fixed sites per kind (empty lists: regenerate anywhere).
        self._empty_since = {}  # Site cell -> time step at which its resource was last found consumed.
        self.active_half: Optional[str] = random.choice(('left', 'right')) if BEACON_CYCLE > 0 else None
        self._next_redraw = 0  # Time step of the next beacon redraw.

        if FOOD_SITES > 0:
            self._place_sites('food', FOOD_SITES)
        else:
            self._place_initial_food()  # Place some food at the start
        if WATER_SITES > 0:
            self._place_sites('water', WATER_SITES)
        else:
            self._place_initial_water()  # Place some water at the start
        self._place_initial_obstacles()  # Place some obstacles at the start

    def _place_initial_food(self):
        """
        Places a few initial food items randomly on the grid.
        Ensures food is placed only on 'empty' cells.
        """
        food_placed = 0
        attempts = 0
        while food_placed < MAX_FOOD_ITEMS / 2 and attempts < GRID_SIZE * GRID_SIZE * 2:  # Limit attempts
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.grid[row][col] == 'empty':
                self.grid[row][col] = 'food'
                food_placed += 1
            attempts += 1
        print(f"World: Placed {food_placed} initial food items.")

    def _place_initial_water(self):
        """
        New: Places a few initial water items randomly on the grid.
        Ensures water is placed only on 'empty' cells.
        """
        water_placed = 0
        attempts = 0
        while water_placed < MAX_WATER_ITEMS / 2 and attempts < GRID_SIZE * GRID_SIZE * 2:  # Limit attempts
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.grid[row][col] == 'empty':
                self.grid[row][col] = 'water'
                water_placed += 1
            attempts += 1
        print(f"World: Placed {water_placed} initial water items.")

    def _place_initial_obstacles(self):
        """
        New: Places a few initial obstacles randomly on the grid.
        Ensures obstacles are placed only on 'empty' cells.
        """
        obstacles_placed = 0
        attempts = 0
        while obstacles_placed < MAX_OBSTACLES and attempts < GRID_SIZE * GRID_SIZE * 2:  # Limit attempts
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if self.grid[row][col] == 'empty':
                self.grid[row][col] = 'obstacle'
                obstacles_placed += 1
            attempts += 1
        print(f"World: Placed {obstacles_placed} initial obstacles.")

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
            # Check if position is valid and not an obstacle
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.grid[row][col] != 'obstacle':
                agent.pos_x = row
                agent.pos_y = col
            else:
                print(
                    f"Warning: Invalid or occupied (by obstacle) start position {start_pos} for agent {agent.name}. Placing randomly.")
                self._place_agent_randomly(agent)
        else:
            self._place_agent_randomly(agent)

        agent.set_environment(self)  # Link the agent back to this world instance.
        self.agents.append(agent)  # Add the agent to the world's list of active agents.

    def _place_agent_randomly(self, agent: Agent):
        """Helper to place an agent at a random empty spot (not obstacle) on the grid."""
        while True:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            # Agents can share a cell with 'empty', 'food', or 'water' but NOT 'obstacle'
            if self.grid[row][col] in ['empty', 'food', 'water']:
                agent.pos_x = row
                agent.pos_y = col
                break

    def _place_sites(self, kind: str, count: int) -> None:
        """Chooses `count` distinct empty cells as the permanent sites of `kind` and fills them."""
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c] == 'empty']
        if BEACON_CYCLE > 0:  # Both halves get sites, so either can be the active one.
            chosen = []
            for half in ('left', 'right'):
                in_half = [cell for cell in cells if half_of(cell[1]) == half]
                chosen += random.sample(in_half, min(count // 2, len(in_half)))
            self._sites[kind] = chosen
        else:
            self._sites[kind] = random.sample(cells, min(count, len(cells)))
        for r, c in self._sites[kind]:
            self.grid[r][c] = kind
        print(f"World: Placed {len(self._sites[kind])} fixed {kind} sites.")

    def _redraw_active_half(self) -> None:
        """Picks the active half afresh and empties every site of the other half."""
        self.active_half = random.choice(('left', 'right'))
        for kind in ('food', 'water'):
            for row, col in self._sites[kind]:
                if half_of(col) != self.active_half and self.grid[row][col] == kind:
                    self.grid[row][col] = 'empty'
                self._empty_since.pop((row, col), None)

    def _regrow_sites(self, kind: str) -> int:
        """Regrows consumed resources on their own sites once SITE_REGROW_STEPS have passed."""
        for cell in self._sites[kind]:
            row, col = cell
            if self.active_half is not None and half_of(col) != self.active_half:
                continue  # Dormant half: nothing grows here.
            if self.grid[row][col] == kind:
                self._empty_since.pop(cell, None)
            elif self.grid[row][col] == 'empty' and not any(a.pos_x == row and a.pos_y == col for a in self.agents):
                since = self._empty_since.setdefault(cell, self.time_step)
                if self.time_step - since >= SITE_REGROW_STEPS:
                    self.grid[row][col] = kind
                    del self._empty_since[cell]
            else:
                self._empty_since.pop(cell, None)  # Blocked by an obstacle or an agent: the clock restarts.
        return sum(row.count(kind) for row in self.grid)

    def _regenerate(self, kind: str, max_items: int, chance: float) -> int:
        """Gives empty cells a chance to grow `kind`, up to `max_items`; returns how many exist afterwards."""
        if self._sites[kind]:
            return self._regrow_sites(kind)
        count = sum(row.count(kind) for row in self.grid)
        if count >= max_items:
            return count
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c] == 'empty']
        if RESPAWN_ORDER == "uniform":
            random.shuffle(cells)
        elif RESPAWN_ORDER != "row_major":
            raise ValueError(f"unknown RESPAWN_ORDER {RESPAWN_ORDER!r}")
        for r, c in cells:
            if random.random() < chance:
                self.grid[r][c] = kind
                count += 1
                if count >= max_items:
                    break
        return count

    def update_environment(self):
        """
        Updates the environmental conditions for the current time step,
        including food/water regeneration, day/night cycle, and weather.
        """
        if BEACON_CYCLE > 0 and self.time_step >= self._next_redraw:
            self._redraw_active_half()
            length = BEACON_CYCLE
            if BEACON_CYCLE_JITTER > 0:
                length = max(1, round(BEACON_CYCLE * random.uniform(1 - BEACON_CYCLE_JITTER, 1 + BEACON_CYCLE_JITTER)))
            self._next_redraw = self.time_step + length
        self.food_available = self._regenerate('food', MAX_FOOD_ITEMS, FOOD_SPAWN_CHANCE) > 0
        self.water_available = self._regenerate('water', MAX_WATER_ITEMS, WATER_SPAWN_CHANCE) > 0

        # Dynamic obstacle spawning (simple random placement)
        current_obstacle_count = sum(row.count('obstacle') for row in self.grid)
        if current_obstacle_count < MAX_OBSTACLES:
            if random.random() < OBSTACLE_SPAWN_CHANCE:
                row, col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                # Ensure no agent or vital resource is on the cell where obstacle spawns
                if self.grid[row][col] == 'empty' and \
                        not any(a.pos_x == row and a.pos_y == col for a in self.agents):
                    self.grid[row][col] = 'obstacle'
                    print(f"World: New obstacle spawned at ({row},{col}).")

        # Simulate a simple day/night cycle.
        if self.time_step % 10 < 5:
            self.time_of_day = "day"
        else:
            self.time_of_day = "night"

        # Simulate simple weather changes
        if self.time_step % 20 == 0:  # Change weather every 20 steps
            self.current_weather = random.choice(WEATHER_TYPES)
            print(f"World: Weather changed to {self.current_weather}.")

    def step(self):
        """
        Executes a single simulation step.

        During each step, the environment is updated, and then each agent
        in the world perceives its surroundings, thinks (decides an action),
        and acts upon that decision. Agent status is logged before actions.
        """
        self.update_environment()  # Update global conditions like food, water, time of day, and weather.
        # Log the current time step and environmental conditions.
        print(
            f"\n=== Time Step {self.time_step} | Food: {self.food_available} | Water: {self.water_available} | Time: {self.time_of_day} | Weather: {self.current_weather} ===")

        # Optional: Print the grid for visualization
        self.print_grid()

        for agent in self.agents:
            # Agent's internal monologue is now handled within agent.think()
            # and printed in main.py.
            agent.sense()  # Agent perceives the updated environment.
            action = agent.think()  # Agent processes perceptions and decides an action.
            # print(f"{agent.name} decided to {action}") # Moved to internal monologue
            agent.act(action)  # Agent performs the chosen action, affecting its internal state.

        self.time_step += 1  # Advance the global simulation time step.

    def run(self, steps: int = 10):
        """
        Runs the simulation for a specified number of steps.

        Args:
            steps (int, optional): The total number of simulation steps to run. Defaults to 10.
        """
        for _ in range(steps):
            self.step()

    def print_grid(self):
        """
        Prints a visual representation of the grid, including agent positions.
        'F' for food, 'W' for water, '#' for obstacle, 'A' for agent, '.' for empty.
        If an agent is on food/water/obstacle, it will show 'A'.
        """
        display_grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Populate food, water, and obstacles
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] == 'food':
                    display_grid[r][c] = 'F'
                elif self.grid[r][c] == 'water':  # Display water
                    display_grid[r][c] = 'W'
                elif self.grid[r][c] == 'obstacle':  # Display obstacle
                    display_grid[r][c] = '#'

        # Place agents on the display grid (agents override other elements for display)
        for agent in self.agents:
            if hasattr(agent, 'pos_x') and hasattr(agent, 'pos_y'):
                row, col = agent.pos_x, agent.pos_y
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    display_grid[row][col] = 'A'
            else:
                print(f"Warning: Agent {agent.name} has no position to display.")

        print("\n--- Current Grid ---")
        for row in display_grid:
            print(" ".join(row))
        print("--------------------")

    def move_object_at_position(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """
        New: Attempts to move an 'obstacle' from one grid position to an adjacent empty one.

        Args:
            from_x (int): The row of the object to move.
            from_y (int): The column of the object to move.
            to_x (int): The target row to move the object to.
            to_y (int): The target column to move the object to.

        Returns:
            bool: True if the object was successfully moved, False otherwise.
        """
        # Check if 'from' coordinates are valid and contain an obstacle
        if not (0 <= from_x < GRID_SIZE and 0 <= from_y < GRID_SIZE and self.grid[from_x][from_y] == 'obstacle'):
            return False  # No obstacle at the source or invalid source coordinates

        # Check if 'to' coordinates are valid and empty
        if not (0 <= to_x < GRID_SIZE and 0 <= to_y < GRID_SIZE and self.grid[to_x][to_y] == 'empty'):
            return False  # Target is invalid or not empty

        # Check if 'to' position is adjacent to 'from' position (Manhattan distance of 1)
        if abs(from_x - to_x) + abs(from_y - to_y) != 1:
            return False  # Not an adjacent cell

        # Perform the move
        self.grid[to_x][to_y] = 'obstacle'
        self.grid[from_x][from_y] = 'empty'
        return True
