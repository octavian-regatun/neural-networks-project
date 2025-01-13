# Flappy Bird DQN Agent

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird using reinforcement learning.

## Architecture

### Neural Network Structure
The DQN uses a Convolutional Neural Network (CNN) architecture:

1. Input Layer
   - Accepts 64x64 grayscale images (preprocessed game frames)
   - Input shape: (1, 64, 64)

2. Convolutional Layers
   - Conv2D: 32 filters, 8x8 kernel, stride 4, ReLU activation
   - Conv2D: 64 filters, 4x4 kernel, stride 2, ReLU activation

3. Fully Connected Layers
   - Dense: 512 nodes, ReLU activation
   - Dense: 512 nodes, ReLU activation
   - Output: 2 nodes (representing Q-values for actions: do nothing/flap)

### Game State Preprocessing
- Frame cropping to remove ground
- HSV color space conversion
- Background element masking (sky, grass, clouds, base)
- Resizing to 64x64
- Normalization to [0,1] range

## Hyperparameters

### DQN Parameters
- Learning rate: 0.00025
- Discount factor (gamma): 0.99
- Replay buffer capacity: 100,000 transitions
- Mini-batch size: 64
- Target network sync rate: 1000 steps

### Exploration Strategy
- Initial epsilon: 1.0
- Minimum epsilon: 0.0001
- Epsilon decay rate: 0.9997

### Reward Structure
- Passing pipe: +1.0
- Staying alive: +0.1
- Game over: -1.0

## Training Process
- Training runs for up to 10,000 episodes
- Early stopping when reward reaches 50,000
- Checkpoints saved every 100 episodes
- Best model saved when achieving new highest reward

## Results
The agent demonstrates successful learning by:
- Consistently navigating through pipes
- Learning to maintain safe altitude
- Developing stable flying patterns

[Training statistics and performance graphs should be added here]