import flappy_bird_gymnasium
import gymnasium as gym
import torch
import numpy as np
import cv2
from agent import DQNAgent  # Add this import
import matplotlib.pyplot as plt
import os  # Add this import at the top
from datetime import datetime

def preprocess_frame(frame):
    # Crop the bottom area (ground)
    frame = frame[:400, :]  # Adjust 400 if needed to match exact ground position
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define masks for background elements
    masks = [
        # Sky (light blue)
        cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255])),
        # Grass (light green)
        cv2.inRange(hsv, np.array([50, 100, 100]), np.array([75, 255, 255])),
        # Clouds (whitish)
        cv2.inRange(hsv, np.array([40, 0, 130]), np.array([100, 120, 255])),
        # Base (brown/beige)
        cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 150, 150]))
    ]
    
    # Combine all background masks
    background_mask = masks[0]
    for mask in masks[1:]:
        background_mask = cv2.bitwise_or(background_mask, mask)
    
    # Invert mask to keep objects of interest (bird and pipes)
    objects_mask = cv2.bitwise_not(background_mask)
    
    # Create result with white objects on black background
    result = np.zeros_like(frame[:,:,0])
    result[objects_mask > 0] = 255
    
    # Resize to 64x64 and normalize
    resized = cv2.resize(result, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    return normalized

def plot_training_stats(episode_rewards, epsilon_values, episodes, save=True):
    # Plot rewards
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.title(f'Training Rewards over {episodes} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot epsilon
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(epsilon_values) * 10, 10), epsilon_values, 'r-')
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    if save:
        plt.savefig('training_stats.png')
    plt.close()

def main():
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Add file logging setup with timestamp
    log_file = open('training_log.txt', 'a')
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n\n=== New Training Session - {start_time} ===\n")

    # Create the environment
    env = gym.make('FlappyBird-v0', render_mode='rgb_array')
    observation, info = env.reset()

    # Create the agent
    state_shape = (64, 64)  # Preprocessed frame dimensions
    n_actions = 2  # Flap or do nothing
    agent = DQNAgent(state_shape, n_actions)

    # Training loop
    episode_rewards = []
    epsilon_values = []
    best_reward = float('-inf')

    for episode in range(10000):  # Number of episodes
        observation, info = env.reset()
        episode_reward = 0
        
        while True:
            frame = env.render()
            processed_frame = preprocess_frame(frame)
            
            # # Display preprocessed frame
            # display_frame = (processed_frame * 255).astype(np.uint8)  # Convert back to 0-255 range
            # display_frame = cv2.resize(display_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('Preprocessed View', display_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Select and perform action
            action = agent.select_action(processed_frame)
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Modify reward according to our reward system
            if terminated:
                reward = -1.0
            elif info.get('score', 0) > 0:
                reward = 1.0
            else:
                reward = 0.1
            
            # Store transition and train
            next_frame = preprocess_frame(env.render())
            agent.memory.push(processed_frame, action, reward, next_frame, terminated)
            agent.train()
            
            episode_reward += reward
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(agent.policy_net.state_dict(), 'checkpoints/best_model.pth')
                
                # Update checkpoint path
                if episode % 100 == 0:
                    checkpoint_path = f'checkpoints/model_checkpoint_ep{episode}.pth'
                    torch.save(agent.policy_net.state_dict(), checkpoint_path)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{timestamp}] Saved checkpoint at episode {episode}\n")
                    log_file.flush()
                break
        
        # Modify the printing and logging section
        if episode % 10 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats = f"[{timestamp}] Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, Epsilon: {agent.epsilon:.5f}"
            print(stats)
            log_file.write(stats + '\n')
            log_file.flush()  # Ensure immediate writing to file
            epsilon_values.append(agent.epsilon)
            plot_training_stats(episode_rewards, epsilon_values, episode + 1)

        if episode_reward >= 50000:  # stop_on_reward
            print(f"Reached reward threshold of 50000! Stopping training.")
            torch.save(agent.policy_net.state_dict(), 'checkpoints/final_model.pth')
            epsilon_values.append(agent.epsilon)
            plot_training_stats(episode_rewards, epsilon_values, episode + 1)  # Final plot
            break

    # Add cleanup at the end
    log_file.close()
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
