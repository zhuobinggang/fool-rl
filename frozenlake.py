import gym
import pg0 as M
import random
import logging
import torch as t
import utils as U

policy = U.policy

G = {
  'm': None, 
  'episode_counter': 0,
  'epoch_rewards': []
}

def run(episode_max = 5000,T = 30):
  U.init_logger('frozenlake.log')
  if G['m'] == None:
    logging.info('Use stored model')
    G['m'] = m = M.FrozenLake_V0()
  else:
    m = G['m']
  env = gym.make('FrozenLake-v0')
  count = 0
  for _ in range(episode_max):
    history = []
    state = env.reset()
    # play the game one episode
    step_counter = 0
    for _ in range(T):
      step_counter += 1
      action = policy(state, env, m)
      (state_next, reward, done, info) = env.step(action)
      history.append((state, action, reward))
      state = state_next
      if done:
        break
      else:
        pass
    # train the model
    reward_sum = sum([reward for (state, action, reward) in history])
    if reward_sum > 0:
      losses = [m.loss(state, action) for (state, action, reward) in history]
      loss = t.stack(losses).sum() * reward_sum
      m.zero_grad()
      loss.backward()
      m.optim.step()
    else:
      pass
    # Logging out info
    G['episode_counter'] += 1
    if reward_sum > 0:
      count += 1
      logging.info(f'episode: {G["episode_counter"]}, reward_sum: {reward_sum}, step_counter: {step_counter}')
  # Record epoch rewards
  G['epoch_rewards'].append(count)


def fuck_out(m):
  for i in range(16):
    print(m.sample(i))

def run_env(m, env):
  state = env.reset()
  for _ in range(100):
    action = m.sample(state)
    (state_next, reward, done, info) = env.step(action)
    state = state_next
    env.render()
    if reward > 0:
      print(f'Got reward {reward}')
    if done:
      break

