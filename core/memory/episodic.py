from collections import deque

class EpisodicMemory:
    """
    Manages a limited capacity episodic memory for an agent.

    This class uses a deque (double-ended queue) to store a fixed number of
    recent experiences (episodes), providing the agent with a short-term
    recollection of past perceptions, internal states, and actions.
    """
    def __init__(self, capacity: int = 5):
        """
        Initializes the episodic memory with a specified capacity.

        Args:
            capacity (int): The maximum number of episodes to store in memory.
                            When new episodes are added and the capacity is reached,
                            the oldest episode is automatically discarded. Defaults to 5.
        """
        self.capacity: int = capacity
        # deque is used for efficient appending and popping from both ends,
        # and 'maxlen' automatically handles capacity limits.
        self.memory: deque = deque(maxlen=capacity)

    def add(self, step: int, perception: dict, state, action: str):
        """
        Adds a new episode (experience) to the agent's episodic memory.

        Each episode captures a snapshot of the agent's situation at a specific
        time step, including its perceptions, internal physiological state, and
        the action it performed.

        Args:
            step (int): The current simulation time step.
            perception (dict): A dictionary representing the agent's sensory
                               input and environmental observations at this step.
            state: The agent's internal state object (e.g., InternalState instance),
                   containing attributes like 'hunger', 'fatigue', and 'mood'.
            action (str): The action performed by the agent at this step.
        """
        episode = {
            "step": step,
            "perception": perception.copy(), # Store a copy to prevent external modifications
            "state": {
                "hunger": round(state.hunger, 2),    # Rounded for cleaner memory representation
                "fatigue": round(state.fatigue, 2),  # Rounded for cleaner memory representation
                "mood": state.mood                   # Store current mood
            },
            "action": action
        }
        self.memory.append(episode)

    def get_memory(self) -> list[dict]:
        """
        Retrieves all currently stored episodes in the episodic memory.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an episode.
                        The episodes are ordered from oldest to newest.
        """
        return list(self.memory)

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the episodic memory content.

        This method is useful for quickly reviewing the agent's recent history
        for logging or debugging purposes.

        Returns:
            str: A multi-line string, with each line detailing a remembered episode's
                 step, action, and mood.
        """
        # Format each episode into a readable string and join them with newlines.
        return "\n".join([f"Step {ep['step']}: {ep['action']}, Mood: {ep['state']['mood']:.2f}" for ep in self.memory])