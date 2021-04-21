import logging
import random

def init_logger(path):
  logging.basicConfig(
    filename=path,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# return: action = 0, 1, 2, 3
def policy(state, env, m, prob = 0.2):
  if random.random() < prob:
    # Expolore
    return env.action_space.sample()
  else: 
    # Sample from model
    return m.sample(state)
