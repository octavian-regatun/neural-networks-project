import flappy_bird_gymnasium
import gymnasium as gym
import torch
import numpy as np
import cv2

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
    
    # Resize to 64x64
    resized = cv2.resize(result, (64, 64), interpolation=cv2.INTER_AREA)
    return resized

# Create the environment
env = gym.make('FlappyBird-v0', render_mode='rgb_array')
observation, info = env.reset()

# Game loop
while True:
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Get frame and preprocess
    frame = env.render()
    processed_frame = preprocess_frame(frame)
    
    # Show both original and processed frames
    cv2.imshow('Flappy Bird Original', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imshow('Flappy Bird Processed', processed_frame)
    
    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Reset if game is over
    if terminated or truncated:
        observation, info = env.reset()

env.close()
cv2.destroyAllWindows()
