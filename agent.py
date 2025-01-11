import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.memory = ReplayBuffer()
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 1000
        self.steps = 0
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        current_q = self.model(state).gather(1, action.unsqueeze(1))
        next_q = self.target_model(next_state).max(1)[0].detach()
        target_q = reward + (1 - done) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
