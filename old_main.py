import time
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Hyperparameters
learning_rate = 0.0001
discount_factor = 0.99
initial_epsilon = 1.0
final_epsilon = 0.1
epsilon_decay = 0.99995
replay_memory_size = 100000
batch_size = 32
agent_history_length = 4
replay_start_size = 2000
update_target_every = 500

def modify_reward(reward, terminated, info):
    if terminated:
        return -100
    elif 'score' in info and info['score'] > 0:
        return 10
    else:
        return 0.3


# Preprocessing function
def preprocess(observation):
    if observation is None:
        return None
    observation = observation[:-108, :, :]  # Crop bottom UI area
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(binary, (80, 80))  # Resize to 80x80
    return resized

# Frame stacking function
def stack_frames(frame_history, new_frame):
    frame_history.append(new_frame)
    if len(frame_history) > agent_history_length:
        frame_history.pop(0)

    if len(frame_history) == agent_history_length:
        stacked_frames = np.stack(frame_history, axis=0)  # Shape: (4, 80, 80)
        return stacked_frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
    else:
        return None

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_input_size = self._get_conv_output((4, 80, 80))

        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training loop
def train_q_network(env):
    num_actions = env.action_space.n
    q_network = QNetwork(num_actions).to(device)
    target_network = QNetwork(num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=replay_memory_size)
    frame_history = deque(maxlen=agent_history_length)
    # Initialize frame history with the first observation
    observation, _ = env.reset()
    frame = env.render()
    initial_frame = preprocess(frame)
    for _ in range(agent_history_length):
        frame_history.append(initial_frame)
    
    total_rewards = 0
    frame_count = 0
    episode_count = 0
    epsilon = initial_epsilon
    start_time = time.time()
    last_best_avg_reward = float('-inf')

    episode_rewards = []
    losses = []
    current_episode_reward = 0

    while frame_count < 100000:
        frame_count += 1
        
        # Get current state
        stacked_frames = np.stack(frame_history, axis=0)
        state = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)

        # Select action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(state).argmax().item()

        # Take action and get new state
        next_observation, reward, terminated, truncated, info = env.step(action)
        modified_reward = modify_reward(reward, terminated, info)
        current_episode_reward += modified_reward

        # Process new frame
        next_frame = preprocess(env.render())
        frame_history.append(next_frame)
        next_stacked_frames = np.stack(list(frame_history), axis=0)

        # Store transition in replay buffer
        replay_buffer.append((stacked_frames, action, modified_reward, next_stacked_frames, terminated))

        if epsilon > final_epsilon:
            epsilon *= epsilon_decay

        # Training step
        if len(replay_buffer) >= replay_start_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            # Compute Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * discount_factor * next_q_values

            # Compute loss and update
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if terminated or truncated:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            print(f"Episode {episode_count} finished with reward: {current_episode_reward}")
            
            # Reset environment and frame history
            observation, _ = env.reset()
            frame = env.render()
            initial_frame = preprocess(frame)
            frame_history.clear()
            for _ in range(agent_history_length):
                frame_history.append(initial_frame)
                
            current_episode_reward = 0

        # Update target network
        if frame_count % update_target_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Logging
        if frame_count % 1000 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            avg_loss = np.mean(losses[-100:]) if losses else 0
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0

            print(f"Frame: {frame_count}, Episode: {episode_count}, Epsilon: {epsilon:.3f}, "
                  f"Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.2f}, "
                  f"Buffer Size: {len(replay_buffer)}, Time: {elapsed_time:.2f}s")

    torch.save(q_network.state_dict(), "q_network_flappybird.pth")
    print("Final model saved.")

    return q_network

# Testing loop
def test_model(env, q_network, num_episodes=10):
    q_network.eval()
    for episode in range(num_episodes):
        observation, _ = env.reset()
        frame_history = deque(maxlen=agent_history_length)
        total_reward = 0
        terminated = False

        while not terminated:
            frame = env.render()
            processed_frame = preprocess(frame)
            stacked_frames = stack_frames(frame_history, processed_frame)
            if stacked_frames is None:
                continue

            state = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_network(state).argmax().item()

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("FlappyBird-v0", render_mode="rgb_array")
print(device)

# Train the Q-network
trained_q_network = train_q_network(env)

# Test the trained Q-network
test_model(env, trained_q_network)

env.close()