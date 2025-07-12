import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import List, Dict

# Assuming DQNLearner is in core.learning.dqn_learner
from core.learning.dqn_learner import DQNLearner


def plot_dqn_q_values(dqn_learner: DQNLearner, agent_name: str = "Agent", resolution: int = 10):
    """
    Plots the Q-values predicted by the DQN for each action across a discretized
    hunger and fatigue state space.

    Args:
        dqn_learner (DQNLearner): The trained DQNLearner instance.
        agent_name (str): The name of the agent, used in the plot title and filename.
        resolution (int): The number of steps for discretizing hunger and fatigue
                          for visualization (e.g., 10 means 10x10 grid).
    """
    hunger_values = np.linspace(0, 1, resolution)
    fatigue_values = np.linspace(0, 1, resolution)

    # Prepare data structure for Q-values
    # q_values_data[action_name][hunger_idx][fatigue_idx]
    q_values_data: Dict[str, np.ndarray] = {
        action: np.zeros((resolution, resolution)) for action in dqn_learner.actions
    }

    # Iterate through the discretized state space and get Q-values from the DQN
    all_q_values = []  # To collect all Q-values for determining min/max range
    for h_idx, hunger in enumerate(hunger_values):
        for f_idx, fatigue in enumerate(fatigue_values):
            state_tensor = dqn_learner.get_state_representation(hunger, fatigue)
            with torch.no_grad():  # No need to calculate gradients for plotting
                q_values_tensor = dqn_learner.policy_net(state_tensor)

            # Convert Q-values tensor to numpy array and store for each action
            q_values_np = q_values_tensor.squeeze(0).numpy()  # Remove batch dimension
            all_q_values.extend(q_values_np.tolist())  # Add to the list for range calculation

            for action_idx, action_name in enumerate(dqn_learner.idx_to_action.values()):
                q_values_data[action_name][h_idx, f_idx] = q_values_np[action_idx]

    # Determine the global min and max Q-values for consistent color scaling
    if all_q_values:
        global_q_min = np.min(all_q_values)
        global_q_max = np.max(all_q_values)
    else:
        global_q_min = 0.0
        global_q_max = 1.0  # Default range if no Q-values collected

    # If all Q-values are the same (e.g., all zeros), adjust range slightly to make colormap visible
    if global_q_min == global_q_max:
        global_q_min -= 0.1
        global_q_max += 0.1

    print(f"DQN Q-Value Plotting: Min Q-Value = {global_q_min:.4f}, Max Q-Value = {global_q_max:.4f}")

    # Create plots for each action
    num_actions = len(dqn_learner.actions)
    # Determine grid size for subplots (e.g., 2x3 for 6 actions)
    rows = int(np.ceil(np.sqrt(num_actions)))
    cols = int(np.ceil(num_actions / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f'{agent_name} - DQN Q-Values Across State Space', fontsize=16)

    for i, action_name in enumerate(dqn_learner.actions):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Create a heatmap
        # Use 'extent' to set the actual hunger/fatigue ranges on the axes
        # Added vmin and vmax for consistent and visible color scaling
        im = ax.imshow(q_values_data[action_name].T, origin='lower',
                       extent=[0, 1, 0, 1], cmap='viridis', aspect='auto',
                       vmin=global_q_min, vmax=global_q_max)  # Set min/max for color scale

        ax.set_title(f'Action: {action_name}')
        ax.set_xlabel('Hunger')
        ax.set_ylabel('Fatigue')

        # Add colorbar for each subplot
        fig.colorbar(im, ax=ax, orientation='vertical', label='Q-Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{agent_name.lower().replace(' ', '_')}_dqn_q_values.png")
    plt.savefig(filename)
    print(f"DQN Q-values plot saved to {filename}")
    plt.close()

