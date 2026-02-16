"""Agent state timeline visualization."""

import os
import matplotlib.pyplot as plt
from config.constants import OUTPUT_DIR, AGENT_STATE_PLOT


class StateTimelinePlotter:
    """Records and plots agent hunger/fatigue/mood over simulation ticks."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.history: dict[str, list] = {
            "step": [],
            "hunger": [],
            "fatigue": [],
            "mood": [],
        }

    def log(self, step: int, hunger: float, fatigue: float, mood: str) -> None:
        """Append a data point."""
        self.history["step"].append(step)
        self.history["hunger"].append(hunger)
        self.history["fatigue"].append(fatigue)
        self.history["mood"].append(mood)

    def plot(self) -> None:
        """Generate and save the timeline chart."""
        steps = self.history["step"]
        hunger = self.history["hunger"]
        fatigue = self.history["fatigue"]

        mood_colors = {
            "happy": "#2ecc71",
            "neutral": "#95a5a6",
            "sad": "#e74c3c",
        }
        mood_color_list = [mood_colors.get(m, "#3498db") for m in self.history["mood"]]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, hunger, label="Hunger", color="#e67e22")
        plt.plot(steps, fatigue, label="Fatigue", color="#3498db")
        plt.scatter(
            steps, [1.05] * len(steps),
            c=mood_color_list, label="Mood", marker="|", s=100,
        )

        plt.title(f"Agent State Over Time: {self.agent_name}")
        plt.xlabel("Simulation Step")
        plt.ylabel("Level")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, AGENT_STATE_PLOT)
        plt.savefig(path)
        plt.close()
        print(f"📊 Saved {path}")