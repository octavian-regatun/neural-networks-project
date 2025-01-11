import numpy as np
from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.stack(state)), 
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(np.stack(next_state)),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)
