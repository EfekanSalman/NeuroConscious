import matplotlib.pyplot as plt
import os

def plot_internal_states(time_steps: list[int], hunger_history: list[float], fatigue_history: list[float], mood_history: list[float], agent_name: str = "Agent"):
    """
    Plots the historical values of an agent's hunger, fatigue, and mood over time.

    Args:
        time_steps (list[int]): A list of time steps (x-axis values).
        hunger_history (list[float]): A list of hunger values corresponding to each time step.
        fatigue_history (list[float]): A list of fatigue values corresponding to each time step.
        mood_history (list[float]): A list of mood values corresponding to each time step.
        agent_name (str): The name of the agent, used in the plot title and filename.
    """
    plt.figure(figsize=(12, 6)) # Set figure size for better readability

    plt.plot(time_steps, hunger_history, label='Hunger', color='red')
    plt.plot(time_steps, fatigue_history, label='Fatigue', color='blue')
    plt.plot(time_steps, mood_history, label='Mood', color='green')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{agent_name} - Internal State History')
    plt.ylim(0, 1) # Assuming hunger, fatigue, mood are normalized between 0 and 1
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True) # Create 'plots' directory if it doesn't exist
    filename = os.path.join(output_dir, f"{agent_name.lower().replace(' ', '_')}_internal_states.png")
    plt.savefig(filename)
    print(f"Internal state history plot saved to {filename}")
    plt.close() # Close the plot to free up memory