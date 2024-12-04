# Full training loop implementation and hierarchical structure integration
import numpy as np
import torch
from high_level_policy import HighLevelPolicy
from low_level_policy import LowLevelPolicy
from replay_buffer import ReplayBuffer

class HierarchicalTrainer:
    def __init__(self, state_dim, action_dim, subgoal_dim, buffer_capacity, high_lr, low_lr):
        # Initialize policies
        self.high_policy = HighLevelPolicy(state_dim, subgoal_dim)
        self.low_policy = LowLevelPolicy(state_dim, action_dim)
        
        # Optimizers
        self.high_optimizer = torch.optim.Adam(self.high_policy.parameters(), lr=high_lr)
        self.low_optimizer = torch.optim.Adam(self.low_policy.parameters(), lr=low_lr)
        
        # Replay buffers
        self.high_replay_buffer = ReplayBuffer(buffer_capacity, state_dim, subgoal_dim)
        self.low_replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        # Loss function
        self.criterion = torch.nn.MSELoss()
    
    def high_level_step(self, state, subgoal, reward, next_state, done):
        self.high_replay_buffer.add(state, subgoal, reward, next_state, done)

    def low_level_step(self, state, action, reward, next_state, done):
        self.low_replay_buffer.add(state, action, reward, next_state, done)
    
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

    def train(self, episodes, batch_size):
        for episode in range(episodes):
            # Example: Training loop logic for high and low level policies
            self.train_high_level(batch_size)
            self.train_low_level(batch_size)


# # Save the training logic to a file in the project directory
# training_logic_file = os.path.join(main_folder_path, "hierarchical_trainer.py")

# with open(training_logic_file, "w") as f:
#     f.write(training_logic_code)

# # Confirmation of successful training logic file creation
# os.listdir(main_folder_path)
