from collections import deque
from typing import Dict, Optional, Set, Tuple

Cell = Tuple[int, int]
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ObstacleMap:
    """
    What the agent has learned about where obstacles are, from its own 3x3 views.

    Cells never seen are not recorded and count as passable when planning. An obstacle that a later
    view shows to be gone (obstacles can be pushed) is dropped.
    """

    def __init__(self):
        self.blocked: Set[Cell] = set()

    def observe(self, position: Cell, view) -> None:
        if len(view or []) != 3 or any(len(row) != 3 for row in view):
            return
        for row in range(3):
            for col in range(3):
                content = view[row][col]
                cell = (position[0] + row - 1, position[1] + col - 1)
                if content == "obstacle":
                    self.blocked.add(cell)
                elif content != "unknown":  # 'unknown' is sensor noise: it says nothing about the cell
                    self.blocked.discard(cell)


def plan_first_step(start: Cell, goal: Cell, blocked: Set[Cell], size: int) -> Optional[Cell]:
    """
    First cell of a shortest path from start to goal (breadth-first, 4-neighbourhood) around the known
    obstacles, or None when no such path exists. The goal itself is entered even if it is recorded as blocked.
    """
    if start == goal:
        return start
    came_from: Dict[Cell, Cell] = {start: start}
    frontier = deque([start])
    while frontier:
        current = frontier.popleft()
        for d_row, d_col in _STEPS:
            cell = (current[0] + d_row, current[1] + d_col)
            if cell in came_from or not (0 <= cell[0] < size and 0 <= cell[1] < size):
                continue
            if cell in blocked and cell != goal:
                continue
            came_from[cell] = current
            if cell == goal:
                while came_from[cell] != start:
                    cell = came_from[cell]
                return cell
            frontier.append(cell)
    return None
