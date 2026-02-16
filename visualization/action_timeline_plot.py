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
from typing import List, Tuple

def plot_action_timeline(time_steps: List[int], actions: List[str], consciousness_states: List[str], agent_name: str = "Agent"):
    """
    Plots a timeline of the agent's actions, color-coded by its consciousness state.

    Args:
        time_steps (List[int]): A list of time steps (x-axis values).
        actions (List[str]): A list of actions performed at each time step.
        consciousness_states (List[str]): A list of consciousness states at each time step.
        agent_name (str): The name of the agent, used in the plot title and filename.
    """
    plt.figure(figsize=(15, 8)) # Wider figure for timeline

    # Define colors for different consciousness states
    state_colors = {
        "Awake": "skyblue",
        "Asleep": "darkblue",
        "Focused": "gold",
        # Add more states if you introduce them
    }

    # Map actions to numerical y-axis values for plotting
    unique_actions = sorted(list(set(actions)))
    action_to_y = {action: i for i, action in enumerate(unique_actions)}
    y_ticks = [action_to_y[action] for action in unique_actions]

    # Create scatter plot for each action, colored by state
    for i, step in enumerate(time_steps):
        action = actions[i]
        state = consciousness_states[i]
        color = state_colors.get(state, "gray") # Default to gray if state not in map

        plt.scatter(step, action_to_y[action], color=color, s=100, alpha=0.7,
                    label=state if state not in plt.gca().get_legend_handles_labels()[1] else "")
        # The label logic ensures each state appears only once in the legend

    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.title(f'{agent_name} - Action Timeline by Consciousness State')
    plt.yticks(y_ticks, unique_actions) # Set y-axis ticks to action names
    plt.ylim(-0.5, len(unique_actions) - 0.5) # Adjust y-axis limits

    # Create a custom legend for consciousness states
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=state,
                                  markerfacecolor=color, markersize=10)
                       for state, color in state_colors.items()]
    plt.legend(handles=legend_elements, title="Consciousness State", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True) # Create 'plots' directory if it doesn't exist
    filename = os.path.join(output_dir, f"{agent_name.lower().replace(' ', '_')}_action_timeline.png")
    plt.savefig(filename)
    print(f"Action timeline plot saved to {filename}")
    plt.close()

