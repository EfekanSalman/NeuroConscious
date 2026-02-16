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
from typing import List

def plot_internal_states(time_steps: List[int], hunger_history: List[float], fatigue_history: List[float], thirst_history: List[float], mood_history: List[float], agent_name: str = "Agent"):
    """
    Plots the historical values of an agent's internal states (hunger, fatigue, thirst, mood) over time.

    Args:
        time_steps (list[int]): A list of time steps (x-axis values).
        hunger_history (list[float]): A list of historical hunger values.
        fatigue_history (list[float]): A list of historical fatigue values.
        thirst_history (list[float]): A list of historical thirst values. # New: Added thirst history
        mood_history (list[float]): A list of historical numerical mood values.
        agent_name (str): The name of the agent, used in the plot title and filename.
    """
    plt.figure(figsize=(12, 7)) # Set figure size for better readability

    plt.plot(time_steps, hunger_history, label='Hunger', color='red')
    plt.plot(time_steps, fatigue_history, label='Fatigue', color='blue')
    plt.plot(time_steps, thirst_history, label='Thirst', color='cyan') # New: Plot thirst history
    plt.plot(time_steps, mood_history, label='Mood Value', color='green')

    plt.xlabel('Time Step')
    plt.ylabel('Value (0-1)')
    plt.title(f'{agent_name} - Internal State History')
    plt.ylim(0, 1) # Assuming all internal state values are normalized between 0 and 1
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True) # Create 'plots' directory if it doesn't exist
    filename = os.path.join(output_dir, f"{agent_name.lower().replace(' ', '_')}_internal_state_history.png")
    plt.savefig(filename)
    print(f"Internal state history plot saved to {filename}")
    plt.close() # Close the plot to free up memory

