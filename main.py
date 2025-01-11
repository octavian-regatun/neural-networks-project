import flappy_bird_gymnasium
import gymnasium as gym
import torch
import numpy as np
import cv2
from agent import DQNAgent
from collections import deque

def preprocess_frame(frame):
    # Just resize and normalize, as frame is already grayscale
    resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    
    if is_new_episode:
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        stacked_frames.append(frame)
        
    return stacked_frames, np.stack(stacked_frames)

def main():
    env = gym.make("FlappyBird-v0", render_mode="human")  # Change this line
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(4, 2, device)  # 4 stacked frames, 2 actions
    
    stacked_frames = None
    episode_rewards = []
    
    for episode in range(1000):
        state, _ = env.reset()
        stacked_frames, state = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        current_score = 0  # Add this line
        done = False
        
        while not done:
            action = agent.act(state)
            
            # Skip frames for same action
            skip_frames = 4 if action == 1 else 1  # Skip more frames when flapping
            accumulated_reward = 0
            
            for frame in range(skip_frames):
                next_state, reward, done, _, _ = env.step(action)
                env.render()  # Add this line to render each frame
                accumulated_reward += reward
                if reward == 1:  # Add this line: increment score when bird passes a pipe
                    current_score += 1
                    print(f"\rCurrent Score: {current_score}", end="")  # Add this line
                if done:
                    break
            
            stacked_frames, next_state_stacked = stack_frames(stacked_frames, next_state, False)
            
            agent.memory.push(state, action, accumulated_reward, next_state_stacked, done)
            agent.update()
            
            state = next_state_stacked
            episode_reward += accumulated_reward
            
            if done:
                episode_rewards.append(episode_reward)
                print(f"\nEpisode: {episode}, Final Score: {current_score}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")
                
                if episode % 100 == 0:
                    torch.save(agent.model.state_dict(), f"flappy_bird_dqn_{episode}.pth")

if __name__ == "__main__":
    main()
