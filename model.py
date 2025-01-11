import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels=4, n_actions=2):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.fc(flat)
