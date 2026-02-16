"""Episodic memory — bounded FIFO memory of past experiences."""

from collections import deque
from copy import deepcopy


class EpisodicMemory:
    """Fixed-capacity memory storing perception/state/action snapshots.

    Old episodes are automatically evicted when capacity is reached (FIFO).
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.memory: deque[dict] = deque(maxlen=capacity)

    def add(self, step: int, perception: dict, state, action: str) -> None:
        """Record a single episode.

        Args:
            step: Simulation tick number.
            perception: Deep-copied perception snapshot.
            state: Internal state (hunger, fatigue, mood extracted).
            action: The action that was taken.
        """
        episode = {
            "step": step,
            "perception": deepcopy(perception),
            "state": {
                "hunger": round(state.hunger, 2),
                "fatigue": round(state.fatigue, 2),
                "mood": state.mood,
            },
            "action": action,
        }
        self.memory.append(episode)

    def get_memory(self) -> list[dict]:
        """Return all episodes as a list."""
        return list(self.memory)

    def __len__(self) -> int:
        return len(self.memory)

    def __str__(self) -> str:
        return "\n".join(
            f"Step {ep['step']}: {ep['action']}, Mood: {ep['state']['mood']}"
            for ep in self.memory
        )
