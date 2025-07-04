import matplotlib.pyplot as plt

class AgentLogger:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.history = {
            "step": [],
            "hunger": [],
            "fatigue": [],
            "mood": []
        }

    def log(self, step: int, hunger: float, fatigue: float, mood: str):
        self.history["step"].append(step)
        self.history["hunger"].append(hunger)
        self.history["fatigue"].append(fatigue)
        self.history["mood"].append(mood)

    def plot(self):
        steps = self.history["step"]
        hunger = self.history["hunger"]
        fatigue = self.history["fatigue"]

        mood_colors = {
            "happy": "green",
            "neutral": "gray",
            "sad": "red"
        }
        mood_color_list = [mood_colors.get(m, "blue") for m in self.history["mood"]]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, hunger, label="Hunger", color="orange")
        plt.plot(steps, fatigue, label="Fatigue", color="blue")
        plt.scatter(steps, [1.05] * len(steps), c=mood_color_list, label="Mood", marker='|', s=100)

        plt.title(f"Agent State Over Time: {self.agent_name}")
        plt.xlabel("Simulation Step")
        plt.ylabel("Level")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("agent_state_plot.png")
        print("Plot saved as agent_state_plot.png")
        #plt.show() " disabled because in an environment that does not support GUI (Linux mint)