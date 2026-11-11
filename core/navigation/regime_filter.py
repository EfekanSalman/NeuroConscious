"""Infers which half of the grid holds resources from what the agent sees, with no cue (BEACON_ACCESS = 'inferred').

The generative model is two hypotheses, 'left' or 'right' half active. A resource seen in a half is decisive evidence
that it is the active one (the dormant half is emptied). A cell seen empty is weak evidence: in the active half a cell
holds a resource with probability r, so an empty cell shifts belief away from its half by -log(1 - r). Belief is a log-odds
`L` (positive = left), it decays toward 0 (the regime may have been redrawn), and r is learned online from prediction errors
(seen - r) on cells in the half currently believed active. The cue is the believed half, or None while |L| < theta.
"""

import math
from typing import Callable, List, Optional

LEFT, RIGHT = "left", "right"
RESOURCES = ("food", "water")


class RegimeFilter:
    def __init__(self, mode: str, theta: float, cap: float, hazard: float, rate: float, rate_lr: float, invert: bool = False, ttl: int = 0):
        self.mode, self.theta, self.cap, self.hazard = mode, theta, cap, hazard
        self.rate, self.rate_lr, self.invert, self.ttl = rate, rate_lr, invert, ttl
        self.age = 0                  # 'last_ttl': steps since the latest sighting
        self.log_odds = 0.0           # 'filter' modes
        self.last: Optional[str] = None   # 'last' mode: the half of the latest sighting
        self.errors = 0.0             # running sum of |prediction error|, for inspection

    def update(self, view: List[List[str]], row: int, col: int, size: int, half_of: Callable[[int, int], str]) -> Optional[str]:
        seen = {LEFT: 0, RIGHT: 0}      # resources seen, by half
        empty = {LEFT: 0, RIGHT: 0}     # informative cells without a resource, by half
        for dr, view_row in enumerate(view):
            for dc, content in enumerate(view_row):
                if content in ("unknown", "boundary", "obstacle", "agent"):
                    continue
                cell_col = col + dc - 1
                if not 0 <= cell_col < size:
                    continue
                half = half_of(cell_col, size)
                if content in RESOURCES:
                    seen[half] += 1
                else:
                    empty[half] += 1
        cue = self._infer(seen, empty)
        if cue is not None and self.invert:
            cue = RIGHT if cue == LEFT else LEFT
        return cue

    def _infer(self, seen, empty) -> Optional[str]:
        if self.mode in ("last", "last_ttl"):
            self.age += 1
            if seen[LEFT] != seen[RIGHT]:
                self.last, self.age = (LEFT if seen[LEFT] > seen[RIGHT] else RIGHT), 0
            if self.mode == "last_ttl" and self.age > self.ttl:
                return None
            return self.last
        self.log_odds *= 1.0 - self.hazard
        believed = LEFT if self.log_odds > 0 else RIGHT
        if abs(self.log_odds) >= self.theta:     # learn the resource rate from cells in the half believed active
            n, k = seen[believed] + empty[believed], seen[believed]
            if n:
                error = k / n - self.rate
                self.errors += abs(error)
                self.rate = min(0.9, max(0.001, self.rate + self.rate_lr * error))
        if seen[LEFT] != seen[RIGHT]:
            self.log_odds = self.cap if seen[LEFT] > seen[RIGHT] else -self.cap
        elif self.mode == "filter":
            shift = -math.log(1.0 - self.rate)
            self.log_odds += shift * empty[RIGHT] - shift * empty[LEFT]   # empty cells argue against their own half
        self.log_odds = max(-self.cap, min(self.cap, self.log_odds))
        if abs(self.log_odds) < self.theta:
            return None
        return LEFT if self.log_odds > 0 else RIGHT
