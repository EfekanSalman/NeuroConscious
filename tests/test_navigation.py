"""Unit tests for path planning. Run from the repository root: python -m unittest discover tests"""

import unittest

from core.navigation import ObstacleMap, plan_first_step


class PlanFirstStepTest(unittest.TestCase):
    def test_walks_straight_when_nothing_is_in_the_way(self):
        self.assertEqual(plan_first_step((0, 0), (0, 3), set(), 5), (0, 1))

    def test_goes_around_a_wall(self):
        wall = {(1, 0), (1, 1), (1, 2)}
        self.assertEqual(plan_first_step((0, 0), (2, 0), wall, 5), (0, 1))

    def test_follows_a_long_detour_a_greedy_agent_would_not(self):
        # A U-shaped trap: the goal is straight down, behind a wall that spans the grid except at the far end.
        wall = {(1, c) for c in range(0, 4)}
        step = plan_first_step((0, 0), (2, 0), wall, 5)
        self.assertEqual(step, (0, 1))

    def test_returns_none_when_walled_off(self):
        wall = {(1, 0), (1, 1), (0, 1)}
        self.assertIsNone(plan_first_step((0, 0), (2, 2), wall, 3))

    def test_enters_the_goal_even_if_it_is_marked_blocked(self):
        self.assertEqual(plan_first_step((0, 0), (0, 1), {(0, 1)}, 3), (0, 1))

    def test_stays_inside_the_grid(self):
        self.assertIsNone(plan_first_step((0, 0), (0, 5), set(), 3))

    def test_already_there(self):
        self.assertEqual(plan_first_step((2, 2), (2, 2), set(), 5), (2, 2))


class ObstacleMapTest(unittest.TestCase):
    @staticmethod
    def view(**cells):
        grid = [["empty"] * 3 for _ in range(3)]
        for name, content in cells.items():
            grid[int(name[1])][int(name[2])] = content
        return grid

    def test_records_obstacles_at_absolute_positions(self):
        known = ObstacleMap()
        known.observe((5, 5), self.view(c00="obstacle", c22="obstacle"))
        self.assertEqual(known.blocked, {(4, 4), (6, 6)})

    def test_forgets_an_obstacle_seen_gone(self):
        known = ObstacleMap()
        known.observe((5, 5), self.view(c00="obstacle"))
        known.observe((5, 5), self.view())
        self.assertEqual(known.blocked, set())

    def test_unknown_cells_change_nothing(self):
        known = ObstacleMap()
        known.observe((5, 5), self.view(c00="obstacle"))
        known.observe((5, 5), self.view(c00="unknown"))
        self.assertEqual(known.blocked, {(4, 4)})

    def test_malformed_view_is_ignored(self):
        known = ObstacleMap()
        known.observe((5, 5), None)
        known.observe((5, 5), [["empty"]])
        self.assertEqual(known.blocked, set())


if __name__ == "__main__":
    unittest.main()
