# Create initial script structure for hierarchical reinforcement learning with SAC

# High-Level Policy: Handles global goals (subgoals like collecting gems or moving to mazes)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HighLevelPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.softmax(self.fc3(x), dim=-1)
        return action

# Low-Level Policy: Handles primitive actions to achieve subgoals

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LowLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Continuous action space
        return action


# Training Loop: Update policies and replay buffers

class Trainer:
    def __init__(self, high_policy, low_policy, high_replay_buffer, low_replay_buffer):
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.high_replay_buffer = high_replay_buffer
        self.low_replay_buffer = low_replay_buffer

    def train_high_level(self, batch_size):
        if self.high_replay_buffer.size < batch_size:
            return
        states, subgoals, rewards, next_states, dones = self.high_replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        subgoals = torch.tensor(subgoals, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).squeeze()
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).squeeze()

        # Predict subgoals and compute loss
        predicted_subgoals = self.high_policy(states)
        loss = self.criterion(predicted_subgoals, subgoals)

        # Update high-level policy
        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()

    def train_low_level(self, batch_size):
        if self.low_replay_buffer.size < batch_size:
            return
        states, actions, rewards, next_states, dones = self.low_replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).squeeze()
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).squeeze()

        # Predict actions and compute loss
        predicted_actions = self.low_policy(states)
        loss = self.criterion(predicted_actions, actions)

        # Update low-level policy
        self.low_optimizer.zero_grad()
        loss.backward()
        self.low_optimizer.step()

    def train(self, episodes,batch_size):
        for episode in range(episodes):
            self.train_high_level(batch_size)
            self.train_low_level(batch_size)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_buffer = np.zeros((capacity, state_dim))
        self.action_buffer = np.zeros((capacity, action_dim))
        self.reward_buffer = np.zeros((capacity, 1))
        self.next_state_buffer = np.zeros((capacity, state_dim))
        self.done_buffer = np.zeros((capacity, 1))
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
            self.done_buffer[indices],
        )