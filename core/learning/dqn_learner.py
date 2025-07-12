import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque  # For replay buffer
import numpy as np  # For state representation conversion
from typing import List, Tuple, Dict


# Define the Deep Q-Network architecture
class DQNetwork(nn.Module):
    """
    A simple feed-forward neural network for approximating Q-values.
    It takes a state representation as input and outputs Q-values for each action.
    """

    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the DQNetwork.

        Args:
            state_size (int): The number of features in the state representation.
            action_size (int): The number of possible actions.
        """
        super(DQNetwork, self).__init__()
        # Define layers: Input -> Hidden1 -> Hidden2 -> Output
        # The choice of layer sizes (e.g., 64, 128) is arbitrary and can be tuned.
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()  # Activation function
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output Q-values for each action.
        """
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


class ReplayBuffer:
    """
    A simple replay buffer to store experiences (state, action, reward, next_state, done).
    It allows for sampling random batches of experiences for training, which helps
    to break correlations between consecutive samples.
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Samples a random batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        if len(self.buffer) < batch_size:
            return []  # Not enough samples to form a batch
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


class DQNLearner:
    """
    Implements a Deep Q-Learning agent using a neural network for Q-value approximation.

    This class replaces the QTableLearner for more complex and continuous state spaces.
    It uses a replay buffer for experience replay and a target network for stability.
    """

    def __init__(self, actions: List[str], state_size: int,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay_rate: float = 0.995,
                 replay_buffer_capacity: int = 10000, batch_size: int = 64,
                 target_update_frequency: int = 100):
        """
        Initializes the DQNLearner.

        Args:
            actions (List[str]): A list of possible actions the agent can take.
            state_size (int): The dimension of the state representation vector.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            epsilon (float, optional): Initial exploration rate. Defaults to 1.0.
            epsilon_min (float, optional): Minimum exploration rate. Defaults to 0.01.
            epsilon_decay_rate (float, optional): Rate at which epsilon decays. Defaults to 0.995.
            replay_buffer_capacity (int, optional): Max capacity of the replay buffer. Defaults to 10000.
            batch_size (int, optional): Size of the batch sampled from replay buffer for training. Defaults to 64.
            target_update_frequency (int, optional): How often to update the target network. Defaults to 100 steps.
        """
        self.actions: List[str] = actions
        self.action_size: int = len(actions)
        self.state_size: int = state_size

        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay_rate: float = epsilon_decay_rate

        # Main Q-network (policy network)
        self.policy_net = DQNetwork(state_size, self.action_size)
        # Target Q-network (for stable Q-value estimation)
        self.target_net = DQNetwork(state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Set target network to evaluation mode (no gradients)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size: int = batch_size
        self.target_update_frequency: int = target_update_frequency
        self.update_count: int = 0  # Counter for target network updates

        # Map action strings to integer indices for the neural network output
        self.action_to_idx = {action: i for i, action in enumerate(actions)}
        self.idx_to_action = {i: action for i, action in enumerate(actions)}

    def get_state_representation(self, hunger: float, fatigue: float) -> torch.Tensor:
        """
        Converts agent's internal state (hunger, fatigue) into a numerical tensor
        suitable for input to the neural network.

        In a more advanced setup, this could also include local grid view,
        emotion states, and other perceptions.

        Args:
            hunger (float): The agent's current hunger level.
            fatigue (float): The agent's current fatigue level.

        Returns:
            torch.Tensor: A 1D tensor representing the state.
        """
        # For now, a simple 2-element vector (hunger, fatigue)
        # Ensure values are within a reasonable range (0.0 to 1.0)
        state_vector = np.array([hunger, fatigue], dtype=np.float32)
        return torch.from_numpy(state_vector).float().unsqueeze(0)  # unsqueeze(0) adds batch dimension

    def choose_action(self, hunger: float, fatigue: float) -> str:
        """
        Selects an action based on the agent's current state using an epsilon-greedy policy.

        With probability `epsilon`, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value predicted by the policy network
        for the current state is chosen (exploitation).

        Args:
            hunger (float): The agent's current hunger level.
            fatigue (float): The agent's current fatigue level.

        Returns:
            str: The chosen action.
        """
        state_tensor = self.get_state_representation(hunger, fatigue)

        if random.random() < self.epsilon:
            # Exploration: Choose a random action index.
            action_idx = random.randrange(self.action_size)
        else:
            # Exploitation: Choose the action with the maximum Q-value.
            with torch.no_grad():  # Disable gradient calculation for inference
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax(1).item()  # Get the index of the max Q-value

        return self.idx_to_action[action_idx]

    def update(self, prev_hunger: float, prev_fatigue: float, action: str, reward: float, next_hunger: float,
               next_fatigue: float):
        """
        Updates the policy network using a batch of experiences from the replay buffer.
        This method also adds the current experience to the replay buffer.

        Args:
            prev_hunger (float): Hunger level before the action was taken.
            prev_fatigue (float): Fatigue level before the action was taken.
            action (str): The action that was performed.
            reward (float): The immediate reward received after performing the action.
            next_hunger (float): Hunger level after the action was taken.
            next_fatigue (float): Fatigue level after the action was taken.
        """
        # Convert states and action to tensors/indices
        state = self.get_state_representation(prev_hunger, prev_fatigue)
        action_idx = self.action_to_idx[action]
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = self.get_state_representation(next_hunger, next_fatigue)
        # For simplicity, 'done' is always False for now in this continuous simulation.
        # In a real RL episode, 'done' would be True if the episode terminates.
        done = False

        # Push the experience to the replay buffer
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

        # If not enough samples in buffer, do not train yet
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)

        # Transpose the batch (from list of tuples to tuple of lists)
        # This creates separate tensors for states, actions, rewards, next_states, and dones
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*experiences)

        # Concatenate into single tensors
        batch_states = torch.cat(batch_states)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1)  # Add dimension for gather
        batch_rewards = torch.cat(batch_rewards)
        batch_next_states = torch.cat(batch_next_states)
        # batch_dones = torch.tensor(batch_dones, dtype=torch.bool) # Not used if 'done' is always False

        # Compute Q-values for current states using the policy network
        # policy_net(batch_states) gives Q-values for all actions for each state in batch
        # .gather(1, batch_actions) selects the Q-value for the action actually taken
        current_q_values = self.policy_net(batch_states).gather(1, batch_actions).squeeze(1)

        # Compute max Q-values for next states using the target network
        # .detach() prevents gradients from flowing back into the target network
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_states).max(1)[0]  # max(1)[0] gets max value across actions

        # Compute the target Q-values (Bellman equation)
        # target_q = reward + gamma * max_future_q * (1 - done)
        # Since 'done' is always False, (1 - done) is always 1.
        target_q_values = batch_rewards + self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()  # Update weights

        # Update the target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.update_count}")

    def save_model(self, filepath: str):
        """
        Saves the policy network's state dictionary to a file.

        Args:
            filepath (str): The path to the file where the model will be saved.
        """
        try:
            torch.save(self.policy_net.state_dict(), filepath)
            print(f"DQN model saved to {filepath}")
        except IOError as e:
            print(f"Error saving DQN model to {filepath}: {e}")

    def load_model(self, filepath: str):
        """
        Loads the policy network's state dictionary from a file.

        Args:
            filepath (str): The path to the file from which the model will be loaded.
        """
        try:
            self.policy_net.load_state_dict(torch.load(filepath))
            self.policy_net.eval()  # Set to evaluation mode after loading
            self.target_net.load_state_dict(self.policy_net.state_dict())  # Sync target net
            self.target_net.eval()
            print(f"DQN model loaded from {filepath}")
        except FileNotFoundError:
            print(f"No DQN model found at {filepath}. Starting with a new model.")
        except Exception as e:
            print(f"Error loading DQN model from {filepath}: {e}. Starting with a new model.")

    def __str__(self) -> str:
        """
        Provides a string representation of the DQNLearner's current state.
        """
        return f"DQNLearner (Epsilon: {self.epsilon:.3f}, Buffer Size: {len(self.replay_buffer)})"

