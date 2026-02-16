"""World — simulation environment and tick orchestrator."""

import random
from typing import List

from agent.base_agent import Agent
from utils.logger import get_logger
from config.constants import (
    FOOD_SPAWN_PROBABILITY,
    DAY_NIGHT_CYCLE_LENGTH,
)

logger = get_logger(__name__)


class World:
    """Manages environment state and drives the agent loop.

    Responsibilities:
        - Day/night cycle
        - Food availability (stochastic)
        - Orchestrates sense → think → act for all agents
    """

    def __init__(self):
        self.agents: List[Agent] = []
        self.time_step: int = 0
        self.food_available: bool = False
        self.time_of_day: str = "day"

    def add_agent(self, agent: Agent) -> None:
        """Register an agent in this world."""
        agent.set_environment(self)
        self.agents.append(agent)

    def update_environment(self) -> None:
        """Advance environment state (food, day/night)."""
        self.food_available = random.random() < FOOD_SPAWN_PROBABILITY

        half_cycle = DAY_NIGHT_CYCLE_LENGTH // 2
        if self.time_step % DAY_NIGHT_CYCLE_LENGTH < half_cycle:
            self.time_of_day = "day"
        else:
            self.time_of_day = "night"

    def step(self) -> None:
        """Single simulation tick: update env → run all agents → advance clock."""
        self.update_environment()

        logger.debug(
            "Step %d | Food: %s | Time: %s",
            self.time_step, self.food_available, self.time_of_day,
        )

        for agent in self.agents:
            agent.sense()
            action = agent.think()
            agent.act(action)

        self.time_step += 1

    def run(self, steps: int = 10) -> None:
        """Run multiple simulation ticks."""
        for _ in range(steps):
            self.step()
