import torch
import torch.nn as nn
import numpy as np  # Add this import

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_nodes=512):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate size of convolution output
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, fc1_nodes),  # fc1_nodes = 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        return self.fc(self.conv(x))
