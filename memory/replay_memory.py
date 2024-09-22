from collections import deque, namedtuple
import random
import torch
import numpy as np
from config.config import DEVICE

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def get_batch(self, batch_size):
        experiences = self.sample(batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).to(DEVICE)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        done_batch = torch.FloatTensor(batch.done).to(DEVICE)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)