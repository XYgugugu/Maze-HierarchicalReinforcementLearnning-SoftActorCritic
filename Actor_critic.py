import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Bound the log_std for numerical stability
        std = log_std.exp()
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(action) * self.max_action  # Rescale action
        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)  # Correction for tanh squashing
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        return q

# Soft Actor-Critic
class SAC:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to("cuda")
        self.critic_1 = Critic(state_dim, action_dim).to("cuda")
        self.critic_2 = Critic(state_dim, action_dim).to("cuda")
        self.target_critic_1 = Critic(state_dim, action_dim).to("cuda")
        self.target_critic_2 = Critic(state_dim, action_dim).to("cuda")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.max_action = max_action

        # Copy parameters to the target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def update(self, replay_buffer, batch_size):
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to("cuda")
        actions = torch.tensor(actions, dtype=torch.float32).to("cuda")
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to("cuda")
        next_states = torch.tensor(next_states, dtype=torch.float32).to("cuda")
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to("cuda")

        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = nn.MSELoss()(q1, target_q)
        critic_2_loss = nn.MSELoss()(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update Actor
        actions, log_probs = self.actor.sample_action(states)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Critics
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
