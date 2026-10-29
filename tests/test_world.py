"""Unit tests for the world's resource regeneration. Run from the repository root: python -m unittest discover tests"""

import contextlib
import io
import random
import unittest
from unittest import mock

from environment import world as world_module
from environment.world import GRID_SIZE, World


def mean_resource_row(kind: str, order: str, steps: int = 300, seeds: int = 5) -> float:
    rows = []
    for seed in range(seeds):
        random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(world_module, "RESPAWN_ORDER", order):
            world = World()
            for _ in range(steps):
                world.update_environment()
                # Consume everything each step, so every item on the grid was just regenerated.
                rows.extend(r for r in range(GRID_SIZE) for c in range(GRID_SIZE) if world.grid[r][c] == kind)
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        if world.grid[r][c] == kind:
                            world.grid[r][c] = "empty"
    return sum(rows) / len(rows)


class RegenerationTests(unittest.TestCase):
    def test_uniform_regeneration_is_not_biased_to_top_rows(self):
        self.assertAlmostEqual(mean_resource_row("food", "uniform"), (GRID_SIZE - 1) / 2, delta=0.4)

    def test_legacy_row_major_regeneration_piles_up_in_top_rows(self):
        self.assertLess(mean_resource_row("food", "row_major"), mean_resource_row("food", "uniform") - 1.5)

    def test_regeneration_never_exceeds_the_cap(self):
        random.seed(0)
        with contextlib.redirect_stdout(io.StringIO()):
            world = World()
            for _ in range(200):
                world.update_environment()
                for kind, cap in (("food", world_module.MAX_FOOD_ITEMS), ("water", world_module.MAX_WATER_ITEMS)):
                    self.assertLessEqual(sum(row.count(kind) for row in world.grid), cap)

    def test_unknown_order_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(world_module, "RESPAWN_ORDER", "diagonal"):
            world = World()
            with self.assertRaises(ValueError):
                world.update_environment()


class FixedSiteTests(unittest.TestCase):
    def make_world(self, sites=3, regrow=5):
        random.seed(1)
        patches = (mock.patch.object(world_module, "FOOD_SITES", sites), mock.patch.object(world_module, "SITE_REGROW_STEPS", regrow))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            return World()

    @staticmethod
    def update(world):
        with contextlib.redirect_stdout(io.StringIO()):
            world.update_environment()

    def test_resources_start_on_the_sites(self):
        world = self.make_world()
        cells = {(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if world.grid[r][c] == "food"}
        self.assertEqual(cells, set(world._sites["food"]))
        self.assertEqual(len(cells), 3)

    def test_consumed_resource_regrows_on_its_own_site_after_the_delay(self):
        world = self.make_world(regrow=5)
        row, col = world._sites["food"][0]
        world.grid[row][col] = "empty"
        for step in range(5):
            world.time_step = step
            self.update(world)
            self.assertEqual(world.grid[row][col], "empty", f"regrew early at step {step}")
        world.time_step = 5
        self.update(world)
        self.assertEqual(world.grid[row][col], "food")

    def test_food_never_appears_off_the_sites(self):
        world = self.make_world(regrow=1)
        sites = set(world._sites["food"])
        for step in range(100):
            world.time_step = step
            for r, c in sites:
                world.grid[r][c] = "empty"
            self.update(world)
            on_grid = {(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if world.grid[r][c] == "food"}
            self.assertTrue(on_grid <= sites)

    def test_water_still_regenerates_anywhere_when_only_food_has_sites(self):
        world = self.make_world()
        self.assertEqual(world._sites["water"], [])


if __name__ == "__main__":
    unittest.main()
