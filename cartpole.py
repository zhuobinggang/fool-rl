import gym
import pg0 as M
import random
import logging
import torch as t
import utils as U
from importlib import reload

G = {
  'm': None, 
  'episode_counter': 0,
  'epoch_rewards': []
}

def run(episode_max = 500,T = 500):
  gamma = 0.99
  U.init_logger('frozenlake.log')
  if G['m'] == None:
    logging.info('Use stored model')
    G['m'] = m = M.CartPole()
  else:
    m = G['m']
  env = gym.make('CartPole-v1')
  count = 0
  for _ in range(episode_max):
    history = []
    state = env.reset()
    # play the game one episode
    step_counter = 0
    for _ in range(T):
      step_counter += 1
      action = U.policy(state, env, m, prob = 0.3)
      (state_next, reward, done, info) = env.step(action)
      history.append((state, action, reward))
      state = state_next
      if done:
        break
      else:
        pass
    # 收益递减
    rewards_raw = [reward for (state, action, reward) in history]
    rewards = []
    R = 0
    for r in rewards_raw[::-1]:
      R = r + gamma * R
      rewards.append(R)
    rewards = list(reversed(rewards))
    # TODO: Scale rewards
    # Train
    losses = [m.loss(state, action, reward) for (state, action, _), reward in zip(history, rewards)]
    loss = t.stack(losses).sum()
    m.zero_grad()
    loss.backward()
    m.optim.step()
    # Logging out info
    reward_sum = sum([reward for (state, action, reward) in history])
    G['episode_counter'] += 1
    count += reward_sum
    logging.info(f'episode: {G["episode_counter"]}, reward_sum: {reward_sum}, step_counter: {step_counter}')
  # Record epoch rewards
  G['epoch_rewards'].append(count)

def draw():
  env = gym.make('CartPole-v1')
  state = env.reset()
  m = G['m']
  for _ in range(5000):
    env.render()
    (state_next, reward, done, info) = env.step(m.sample(state)) # take a random action
    if done:
      break
    else:
      state = state_next
  env.close()


def reset():
  G['m'] = None
  G['epoch_rewards'] = []
  G['episode_counter'] = 0
