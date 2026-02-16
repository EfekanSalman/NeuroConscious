#!/usr/bin/env python3
#
# Copyright (c) 2025 Efekan Salman
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        "joy": "green",
        "fear": "purple",
        "frustration": "orange",
        "curiosity": "blue"
    }

    for emotion_name, history_data in emotion_history.items():
        if history_data: # Only plot if there's data for this emotion
            # Use .get() with the lowercase emotion_name to retrieve the correct color
            plt.plot(time_steps, history_data, label=emotion_name.capitalize(), color=colors.get(emotion_name, 'gray'))

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

