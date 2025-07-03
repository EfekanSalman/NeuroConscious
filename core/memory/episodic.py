from collections import deque

class EpisodicMemory:
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, step, perception, state, action):
        episode = {
            "step": step,
            "perception": perception.copy(),
            "state": {
                "hunger": round(state.hunger, 2),
                "fatigue": round(state.fatigue, 2),
                "mood": state.mood
            },
            "action": action
        }
        self.memory.append(episode)

    def get_memory(self):
        return list(self.memory)

    def __str__(self):
        return "\n".join([f"Step {ep['step']}: {ep['action']}, Mood: {ep['state']['mood']}" for ep in self.memory])
