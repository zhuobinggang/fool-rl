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

GAMMA = 0.99

env = gym.make("LunarLander-v2")


rew_buffer = deque([0.0], maxlen=10)
rew_buffer2 = []

def render_test():
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()

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
        self.tar_net = nn.Sequential(
                nn.Linear(8, 64),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
        )
        self.tar_net.load_state_dict(self.net.state_dict())
        # self.optim = optim.AdamW(self.net.parameters(), lr = 1e-3)
        self.optim = optim.Adam(self.net.parameters(), lr = 1e-3)
    # obs: (8)
    def forward(self, obs):
        return self.net(torch.tensor(obs))


def episode_random():
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs_new, reward, done, info = env.step(action)
        transitions.append((obs, action, obs_new, reward, done))
        obs = obs_new
        # env.render()
    return transitions


def episode_greed(m):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = m(obs).argmax().item()
        obs_new, reward, done, info = env.step(action)
        transitions.append((obs, action, obs_new, reward, done))
        obs = obs_new
        # env.render()
    return transitions

def play(m, epsilon, render = False):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = m(obs).argmax().item()
        obs_new, reward, done, info = env.step(action)
        transitions.append((obs, action, obs_new, reward, done))
        obs = obs_new
        if render:
            env.render()
    return transitions

def reward_guess_by_observation_and_action(m, obs, action):
    action_reward_guess = m(torch.tensor(obs)) # (4)
    x = action_reward_guess[action] # ()
    return x




def train(m, transitions):
    loss_sum = 0
    for obs, action, obs_next, reward, done in transitions:
        x = reward_guess_by_observation_and_action(m.net, obs, action) # ()
        y = reward + GAMMA * (1 - done) * max_reward_guess_by_observation(m.tar_net, obs_next)
        loss = nn.functional.smooth_l1_loss(x, y)
        loss_sum += loss
        # backprop
        m.zero_grad()
        loss.backward()
        m.optim.step()
    return loss_sum.detach()


def train_v2(m, timesteps):
    reward_per_episode = []
    loss_per_episode = []
    for step in range(timesteps):
        if step % 10 == 0:
            print(f'{step}/{timesteps}')
        epsilon = np.interp(step, [0, timesteps], [1.0, 0.2])
        transitions = episode_greed_with_epsilon(m, epsilon)
        loss_per_episode.append(train(m, transitions))
        reward_per_episode.append(sum([reward for _, _, _, reward, _ in transitions]))
    return reward_per_episode, loss_per_episode

def cal_epsilon(start, end, step, total_steps, end_fraction):
    progress = step / total_steps
    if progress > end_fraction:
        return end
    else:
        return start + progress * (end - start) / end_fraction


def max_reward_guess_by_observation(m, obs_next):
    return m(torch.tensor(obs_next)).max().detach()

def train_one_step(m, transition):
    obs, action, obs_next, reward, done = transition
    x = reward_guess_by_observation_and_action(m.net, obs, action) # ()
    with torch.no_grad():
        y = reward + GAMMA * (1 - done) * max_reward_guess_by_observation(m.tar_net, obs_next)
    loss = nn.functional.smooth_l1_loss(x, y)
    # torch.clamp(loss, min=-1, max=1)
    # backprop
    m.net.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(m.net.parameters(), 10)
    m.optim.step()
    # TODO: copy to targetnet
    m.tar_net.load_state_dict(m.net.state_dict())
    return loss.item()
        
def train_by_timesteps(m, timesteps, start, end, end_fraction = 0.5):
    obs = env.reset()
    episode_rew = 0
    reward2log = []
    reward_per_episode = []
    loss_per_episode = []
    loss_episode = 0
    for step in range(timesteps):
        epsilon = cal_epsilon(start, end, step, timesteps, end_fraction)
        # print(epsilon)
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = m(obs).argmax().item()
        obs_new, reward, done, info = env.step(action)
        # env.render()
        # print(epsilon)
        episode_rew += reward
        loss = train_one_step(m, (obs, action, obs_new, reward, done))
        loss_episode += loss
        if done:
            rew_buffer.append(episode_rew)
            rew_buffer2.append(episode_rew)
            reward_per_episode.append(episode_rew)
            episode_rew = 0
            obs = env.reset()
            loss_per_episode.append(loss_episode)
            loss_episode = 0
        else:
            obs = obs_new
        if step % 10 == 0:
            dd = np.mean(rew_buffer)
            reward2log.append(dd)
    return reward2log
    # return reward_per_episode
    # return loss_per_episode


# m = DQN_FOOL()
def train_and_draw(m, path = 'dd.png'):
    rews = train_by_timesteps(m, timesteps = 100000, start = 1, end = 0.05, end_fraction = 0.5)
    x = [i * 10 for i in range(len(rews))]
    plt.ylim([-300, 300])
    plt.plot(x, rews)
    plt.savefig(path)
    _ = play(m, 0, True)
    return rews


def plot_episode_rewards(rews, path):
    plt.clf()
    x = [i for i in range(len(rews))]
    plt.plot(x, rews)
    plt.savefig(path)




