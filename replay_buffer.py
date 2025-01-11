import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.uint8))
    
    def __len__(self):
        return len(self.buffer)
