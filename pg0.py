import torch as t
from itertools import chain
nn = t.nn
import logging
import time
from importlib import reload
import requests
import random
import torch.optim as optim
import numpy as np


class FrozenLake_V0(nn.Module):
  def __init__(self):
    super().__init__()
    self.state_ember = nn.Embedding(20, 5)
    self.main = nn.Sequential(
      nn.Linear(5, 128), 
      nn.Dropout(p = 0.6), # Using dropout will significantly improve the performance of our policy
      nn.Linear(128, 4),
    )
    self.optim = optim.AdamW(self.get_should_update(), lr = 1e-3)
    self.CEL = nn.CrossEntropyLoss()

  def get_should_update(self):
    return chain(self.main.parameters(), self.state_ember.parameters())
   
  # action: 0, 1, 2, 3
  # state: 0 ~ 15
  def loss(self, state, action):
    state = self.state_ember(t.LongTensor([state])) # (1, 5)
    o = self.main(state) # (1, 4)
    label = t.LongTensor([action]) # (1)
    loss = self.CEL(o, label)
    return loss

  # return: 0, 1, 2, 3
  def sample(self, state): 
    state = self.state_ember(t.LongTensor([state])) # (1, 5)
    o = self.main(state) # (1, 4)
    return o.argmax().item()


class CartPole(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
      nn.Linear(4, 128), 
      nn.Dropout(p = 0.6), # Using dropout will significantly improve the performance of our policy
      nn.Linear(128, 2),
    )
    self.baseline = nn.Sequential(
      nn.Linear(4, 64), 
      nn.Linear(64, 1),
    )
    self.optim = optim.AdamW(self.get_should_update(), lr = 1e-2)
    self.CEL = nn.CrossEntropyLoss()
    self.MSE = nn.MSELoss()

  def get_should_update(self):
    return chain(self.main.parameters(), self.baseline.parameters())
   
  # action: 0, 1
  # state: (4)
  def loss(self, state, action, reward):
    state = t.tensor(state).float() # (4)
    o = self.main(state).view(1, 2) # (1, 2)
    assert action == 0 or action == 1
    label = t.LongTensor([action]) # (1)
    loss_main = reward * self.CEL(o, label)
    # print(reward)
    # baseline = self.baseline(state) # (1)
    # loss_main = loss_main * (reward - baseline.detach())
    # loss_baseline = self.MSE(baseline, t.tensor([reward]))
    # loss = loss_main + loss_baseline
    loss = loss_main
    return loss.view(1)

  # return: 0, 1
  def sample(self, state): 
    o = self.main(t.tensor(state).float()).view(1, 2) # (1, 2)
    return o.argmax().item()

