
# Flappy Bird DQN Agent

## Architecture

- Input: 4 stacked grayscale frames (84x84)
- CNN layers:
  - Conv2d(4, 32, kernel=8, stride=4)
  - Conv2d(32, 64, kernel=4, stride=2)
  - Conv2d(64, 64, kernel=3, stride=1)
- FC layers:
  - Linear(64 * 7 * 7, 512)
  - Linear(512, 2)

## Hyperparameters

- Learning rate: 0.00025
- Batch size: 32
- Gamma (discount factor): 0.99
- Epsilon start: 1.0
- Epsilon min: 0.01
- Epsilon decay: 0.995
- Replay buffer size: 100,000
- Target network update frequency: 1000 steps
- Frame skip: 4 frames on flap, 1 frame otherwise

## Optimizations

1. Frame stacking: Using 4 consecutive frames to capture motion
2. Frame skipping: Reducing unnecessary computations
3. Reward accumulation during skipped frames
4. Experience replay for stable learning
5. Target network for stable Q-value updates
6. Epsilon-greedy exploration with decay

## Results

[Add your experimental results and graphs here]