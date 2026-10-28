from dataclasses import dataclass
from typing import List, Optional, Tuple

# What a sighting is worth before the agent's current needs are taken into account.
_BASE_RELEVANCE = {"food": 0.15, "water": 0.15, "obstacle": 0.10, "agent": 0.05}
# Steps after which a remembered sighting is worth half as much (resources get consumed or move on).
HALF_LIFE = 100.0
# Need-independent worth for the 'relevance' policy: resources are equally worth keeping, since
# whichever need is not pressing now will be soon; distractors are worth little.
_KIND_VALUE = {"food": 1.0, "water": 1.0, "obstacle": 0.1, "agent": 0.05}


@dataclass
class Sighting:
    kind: str
    location: Tuple[int, int]
    seen: int


class WorkspaceMemory:
    """
    Sustained workspace content: what the agent keeps in mind about where things are.

    It holds at most `slots` sightings, so new percepts compete for entry. A new sighting
    displaces the least valuable stored one only if it is worth more; worth fades with age.
    The policies differ in what worth means:
      'salience'  - follows the agent's current needs (a hungry agent values food most);
      'relevance' - depends on the kind only: resources outrank obstacles and other agents;
      'fifo'      - no competition: the oldest sighting is evicted, as a plain buffer of the
                    same size would. It is the capacity-matched control.

    Recall is a separate choice: 'nearest' returns the closest remembered location, 'recent'
    the most recently seen one (what the agent's FIFO working memory does).

    Args:
        slots (int): Capacity, in sightings.
        policy (str): 'salience', 'relevance' or 'fifo'.
        recall (str): 'nearest' or 'recent'.
    """

    def __init__(self, slots: int, policy: str = "salience", recall: str = "nearest"):
        if slots < 1:
            raise ValueError("slots must be at least 1")
        if policy not in ("salience", "relevance", "fifo"):
            raise ValueError(f"unknown policy {policy!r}")
        if recall not in ("nearest", "recent"):
            raise ValueError(f"unknown recall rule {recall!r}")
        self.slots = slots
        self.policy = policy
        self.recall_rule = recall
        self.sightings: List[Sighting] = []

    def worth(self, sighting: Sighting, internal_state, now: int) -> float:
        if self.policy == "relevance":
            value = _KIND_VALUE[sighting.kind]
        else:
            need = {"food": internal_state.hunger, "water": internal_state.thirst}.get(sighting.kind, 0.0)
            value = _BASE_RELEVANCE[sighting.kind] + need
        return value * 0.5 ** ((now - sighting.seen) / HALF_LIFE)

    def observe(self, agent) -> None:
        """Offers everything in the agent's current view to the memory."""
        now = agent.current_time_step
        for kind, location in self._visible(agent):
            self._offer(Sighting(kind, location, now), agent.internal_state, now)

    def recall(self, kind: str, agent) -> Optional[Tuple[int, int]]:
        """The nearest remembered location of a resource, forgetting any the current view contradicts."""
        view = agent.perception.get("local_grid_view") or []
        here = (agent.pos_x, agent.pos_y)
        best, best_key = None, None
        for sighting in list(self.sightings):
            if sighting.kind != kind:
                continue
            d_row, d_col = sighting.location[0] - here[0], sighting.location[1] - here[1]
            in_view = abs(d_row) <= 1 and abs(d_col) <= 1 and len(view) == 3 and len(view[1 + d_row]) == 3
            if in_view and view[1 + d_row][1 + d_col] not in (kind, "unknown"):
                self.sightings.remove(sighting)  # visible and gone
                continue
            key = ((abs(d_row) + abs(d_col)), -sighting.seen) if self.recall_rule == "nearest" else (-sighting.seen,)
            if best_key is None or key < best_key:
                best, best_key = sighting.location, key
        return best

    def forget(self, kind: str, location: Tuple[int, int]) -> None:
        """Drops a sighting, e.g. because it cannot be reached."""
        self.sightings = [s for s in self.sightings if not (s.kind == kind and s.location == location)]

    @staticmethod
    def _visible(agent):
        """(kind, location) for every resource, obstacle and other agent the agent currently perceives."""
        view = agent.perception.get("local_grid_view") or []
        if len(view) == 3 and all(len(row) == 3 for row in view):
            for row in range(3):
                for col in range(3):
                    if view[row][col] in ("food", "water", "obstacle"):
                        yield view[row][col], (agent.pos_x + row - 1, agent.pos_y + col - 1)
        for other in agent.perception.get("other_agents_in_sight") or []:
            yield "agent", (other["pos_x"], other["pos_y"])

    def _offer(self, new: Sighting, internal_state, now: int) -> None:
        for stored in self.sightings:
            if stored.kind == new.kind and stored.location == new.location:
                stored.seen = new.seen  # Seen again: refresh rather than duplicate.
                return
        if len(self.sightings) < self.slots:
            self.sightings.append(new)
            return
        if self.policy == "fifo":
            self.sightings.remove(min(self.sightings, key=lambda s: s.seen))
            self.sightings.append(new)
            return
        weakest = min(self.sightings, key=lambda s: self.worth(s, internal_state, now))
        if self.worth(new, internal_state, now) > self.worth(weakest, internal_state, now):
            self.sightings.remove(weakest)
            self.sightings.append(new)
