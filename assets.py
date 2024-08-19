import math
import random
from collections import namedtuple, deque
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, n_observations, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.net(x)

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0

    def push(self, state, action, value, next_state, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.length += 1

    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0

def normalize_rewards(rewards, gamma):
    normalized_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        normalized_rewards.insert(0, R)
    return normalized_rewards

class A2C:
    def __init__(self, state_size, action_size, gamma, lr_actor, lr_critic, tau, entropy_coef, clip_grad, hidden_units, batch_size, log_dir):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, action_size, hidden_units).to(device)
        self.critic = Critic(state_size, hidden_units).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)
        
        self.gamma = gamma
        self.tau = tau
        self.episode_rewards = []
        self.memory = Memory()
        self.entropy_coef = entropy_coef
        self.clip_grad = clip_grad
        self.batch_size = batch_size
        
        # TensorBoard setup
        self.writer = SummaryWriter(log_dir=log_dir)
        
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label='Total Reward')
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, num_samples=1)
            value = self.critic(state)
        return action, value
    
    def plot_rewards(self):
        self.reward_line.set_xdata(range(len(self.episode_rewards)))
        self.reward_line.set_ydata(self.episode_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self):
        states = torch.cat(self.memory.states)
        actions = torch.cat(self.memory.actions)
        values = torch.cat(self.memory.values)
        next_states = torch.cat([s if s is not None else torch.zeros_like(states[:1]) for s in self.memory.next_states])
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32, device=device)

        normalized_rewards = torch.tensor(normalize_rewards(self.memory.rewards, self.gamma), dtype=torch.float32, device=device)

        td_target = normalized_rewards + self.gamma * self.critic(next_states).squeeze(1)
        td_error = td_target - values.squeeze(1)
        critic_loss = 0.5 * td_error.pow(2).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer_critic.step()

        advantage = td_error.detach()
        action_probs = self.actor(states)
        log_action_probs = torch.log(action_probs + 1e-10)
        actor_loss = -(log_action_probs[range(len(actions)), actions.squeeze(1)] * advantage).mean()

        entropy = -(action_probs * log_action_probs).sum(1).mean()
        actor_loss -= self.entropy_coef * entropy

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optimizer_actor.step()

        # TensorBoard logging
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), len(self.episode_rewards))
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), len(self.episode_rewards))
        self.writer.add_scalar('Entropy', entropy.item(), len(self.episode_rewards))
        self.writer.add_scalar('Entropy Coefficient', self.entropy_coef, len(self.episode_rewards))

        self.entropy_coef = max(0.01, self.entropy_coef * 0.995)

        self.memory.clear()

    def close(self):
        self.writer.close()
