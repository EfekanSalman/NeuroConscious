import matplotlib.pyplot as plt
import os

def plot_emotion_history(time_steps: list[int], emotion_history: dict[str, list[float]], agent_name: str = "Agent"):
    """
    Plots the historical values of an agent's emotions (Joy, Fear, Frustration, Curiosity) over time.

    Args:
        time_steps (list[int]): A list of time steps (x-axis values).
        emotion_history (dict[str, list[float]]): A dictionary where keys are emotion names
                                                  (e.g., "Joy", "Fear") and values are lists
                                                  of their historical values.
        agent_name (str): The name of the agent, used in the plot title and filename.
    """
    plt.figure(figsize=(12, 7)) # Set figure size for better readability

    colors = {
        "Joy": "green",
        "Fear": "purple",
        "Frustration": "orange",
        "Curiosity": "blue"
    }

    for emotion_name, history_data in emotion_history.items():
        if history_data: # Only plot if there's data for this emotion
            plt.plot(time_steps, history_data, label=emotion_name, color=colors.get(emotion_name, 'gray'))

    plt.xlabel('Time Step')
    plt.ylabel('Emotion Value')
    plt.title(f'{agent_name} - Emotion History')
    plt.ylim(0, 1) # Assuming emotion values are normalized between 0 and 1
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True) # Create 'plots' directory if it doesn't exist
    filename = os.path.join(output_dir, f"{agent_name.lower().replace(' ', '_')}_emotion_history.png")
    plt.savefig(filename)
    print(f"Emotion history plot saved to {filename}")
    plt.close() # Close the plot to free up memory

