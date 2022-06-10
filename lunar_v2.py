# About pytorch
from torch import nn
import torch
import gym
# import itertools
import numpy as np
import random
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# import gym

class DQN_FOOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(8, 64),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
        )
        self.optim = optim.Adam(self.net.parameters(), lr = 1e-3)
        self.GAMMA = 0.99
        self.env = gym.make("LunarLander-v2")
        self.rew_buffer = deque([0.0], maxlen=10)
        self.reward2log = []
    # obs: (8)
    def forward(self, obs):
        return self.net(torch.tensor(obs))


def cal_epsilon(start, end, step, total_steps, end_fraction):
    progress = step / total_steps
    if progress > end_fraction:
        return end
    else:
        return start + progress * (end - start) / end_fraction

def train_by_timesteps(m, timesteps, start, end, end_fraction = 0.5):
    env = m.env
    obs = env.reset()
    episode_rew = 0
    for current_step in range(timesteps):
        epsilon = cal_epsilon(start, end, current_step, timesteps, end_fraction)
        action = env.action_space.sample() if random.random() <= epsilon else m(obs).argmax().item()
        obs_next, reward, done, _ = env.step(action)
        episode_rew += reward
        x = m(obs)[action]
        with torch.no_grad():
            y = reward + m.GAMMA * (1 - done) * m(obs_next).max()
        loss = nn.functional.smooth_l1_loss(x, y)
        m.zero_grad()
        loss.backward()
        m.optim.step()
        if done:
            obs = env.reset()
            m.rew_buffer.append(episode_rew)
            episode_rew = 0
        else:
            obs = obs_next
        if current_step % 10 == 0:
            m.reward2log.append(np.mean(m.rew_buffer))


def play(m, epsilon = 0.1):
    env = m.env
    obs = env.reset()
    while True:
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = m(obs).argmax().item()
        obs_next, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        else:
            obs = obs_next
        env.render()


def train_and_draw(m, path = 'dd.png'):
    train_by_timesteps(m, timesteps = 100000, start = 1, end = 0.05, end_fraction = 0.5)
    rews = m.reward2log
    x = [i * 10 for i in range(len(rews))]
    plt.ylim([-300, 300])
    plt.plot(x, rews)
    plt.savefig(path)
