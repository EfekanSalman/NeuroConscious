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
}
