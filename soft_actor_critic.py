import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(),
            nn.ReLU()
        )
        self.action_logits = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.layer1(state)
        logits = self.action_logits(x)
        return logits

    def sample_action(self, state, gradient=True):
        logits = self.forward(state)
        if gradient:
            # Use Gumbel-Softmax for differentiable sampling
            action_probs = torch.nn.functional.gumbel_softmax(logits, tau=0.5, hard=True)
            action = action_probs.argmax(dim=-1)
            # Correct log_prob using original logits
            log_prob = torch.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            # Standard sampling without gradients
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

# Critic Network (unchanged)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        if action.dim() > 1 and action.size(-1) == 1:
            action = action.squeeze(-1)
        action_one_hot = nn.functional.one_hot(action.to(torch.int64), num_classes=self.action_dim).float()
        x = torch.cat([state, action_one_hot], dim=1)
        return self.layers(x)

# Soft Actor-Critic
class SAC:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Initialize target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            # Use gradient=False for target action sampling
            next_actions, next_log_probs = self.actor.sample_action(next_states, gradient=False)
            next_actions = next_actions.unsqueeze(1)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs.unsqueeze(1))

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_1_loss = nn.MSELoss()(current_q1, target_q)
        critic_2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)

        # Actor update
        actions_new, log_probs = self.actor.sample_action(states)
        actions_new = actions_new.unsqueeze(1)
        q1_new = self.critic_1(states, actions_new)
        q2_new = self.critic_2(states, actions_new)
        actor_loss = (self.alpha * log_probs.unsqueeze(1) - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item()