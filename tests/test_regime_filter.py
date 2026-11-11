import unittest

from core.navigation.regime_filter import RegimeFilter
from core.navigation.regions import half_of

SIZE = 20


def view(left=(), right=(), center="empty"):
    """A 3x3 view: `left`/`right` list the contents of the cells in the left and right columns of the view."""
    return [[left[r] if r < len(left) else "empty", center, right[r] if r < len(right) else "empty"] for r in range(3)]


def make(mode="filter", **kw):
    args = dict(theta=1.0, cap=2.0, hazard=0.0, rate=0.1, rate_lr=0.0, invert=False)
    args.update(kw)
    return RegimeFilter(mode, **args)


class RegimeFilterTest(unittest.TestCase):
    def step(self, f, v, col=9):
        # column 9 is the last of the left half: the view's left/centre columns are left, its right column is right
        return f.update(v, 5, col, SIZE, half_of)

    def test_a_sighting_is_decisive_and_held_by_the_last_mode(self):
        f = make("last")
        self.assertIsNone(self.step(f, view()))
        self.assertEqual(self.step(f, view(left=("food",))), "left")
        self.assertEqual(self.step(f, view()), "left")
        self.assertEqual(self.step(f, view(right=("water",))), "right")

    def test_empty_cells_move_the_filter_but_not_the_sightings_only_variant(self):
        f, g = make("filter"), make("filter_sightings")
        for filt in (f, g):
            self.step(filt, view(left=("food",)))
        for _ in range(30):
            cues = (self.step(f, view()), self.step(g, view()))
        self.assertEqual(cues[1], "left")      # sightings only: nothing argues against the belief
        self.assertNotEqual(cues[0], "left")   # empty left cells wore the belief down

    def test_unknown_and_boundary_cells_carry_no_evidence(self):
        f = make("filter")
        self.step(f, view(left=("food",)))
        for _ in range(30):
            self.step(f, [["unknown"] * 3, ["boundary"] * 3, ["unknown"] * 3])
        self.assertEqual(self.step(f, [["unknown"] * 3] * 3), "left")

    def test_the_resource_rate_is_learned_from_prediction_errors(self):
        f = make("filter", rate=0.5, rate_lr=0.5)
        self.step(f, view(left=("food",)))
        for _ in range(10):
            self.step(f, view(left=("food", "empty", "empty")))
        self.assertLess(f.rate, 0.5)

    def test_invert_flips_the_cue(self):
        self.assertEqual(self.step(make("last", invert=True), view(left=("food",))), "right")
