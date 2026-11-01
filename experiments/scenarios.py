"""Named bundles of parameter overrides for experiments.run (--scenario NAME)."""

from typing import Any, Dict

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # The default 10x10 world.
    "default": {},
    # A 20x20 world with many resources and many obstacles. With a 3x3 view and a five-item working
    # memory, obstacles and other percepts push resource locations out of memory, so what an agent
    # keeps in mind decides how quickly it finds food and water.
    "distractors": {
        "environment.world:GRID_SIZE": 20,
        "environment.world:MAX_FOOD_ITEMS": 8,
        "environment.world:MAX_WATER_ITEMS": 8,
        "environment.world:MAX_OBSTACLES": 15,
        "environment.world:OBSTACLE_SPAWN_CHANCE": 0.05,
    },
    # The distractors world, but food and water only grow on 6 fixed sites each and regrow in place after
    # 50 steps, so a remembered location stays valid. Tests whether memory has any value once it can.
    "sites": {
        "environment.world:GRID_SIZE": 20,
        "environment.world:MAX_OBSTACLES": 15,
        "environment.world:OBSTACLE_SPAWN_CHANCE": 0.05,
        "environment.world:FOOD_SITES": 6,
        "environment.world:WATER_SITES": 6,
        "environment.world:SITE_REGROW_STEPS": 50,
    },
    # The sites world with the beacon regime: only one half of the grid holds resources, redrawn every 150 steps.
    # Where the resources are is announced only by the cue; see `agent.base_agent:BEACON_ACCESS`.
    "beacon": {
        "environment.world:GRID_SIZE": 20,
        "environment.world:MAX_OBSTACLES": 15,
        "environment.world:OBSTACLE_SPAWN_CHANCE": 0.05,
        "environment.world:FOOD_SITES": 6,
        "environment.world:WATER_SITES": 6,
        "environment.world:SITE_REGROW_STEPS": 50,
        "environment.world:BEACON_CYCLE": 150,
    },
}
