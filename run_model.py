import flappy_bird_gymnasium
import gymnasium as gym
import torch
import cv2
import numpy as np
from model import DQN
from main import preprocess_frame  # Reusing the preprocess_frame function

def run_model():
    # Initialize environment
    env = gym.make('FlappyBird-v0', render_mode='rgb_array')
    
    # Initialize model
    state_shape = (64, 64)
    n_actions = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and load the trained model
    model = DQN(state_shape, n_actions).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()  # Set to evaluation mode
    
    while True:
        observation, info = env.reset()
        total_reward = 0
        
        while True:
            # Get and process the frame
            frame = env.render()
            processed_frame = preprocess_frame(frame)
            
            # Display both original and preprocessed frames
            cv2.imshow('Original Game', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            display_frame = (processed_frame * 255).astype(np.uint8)
            display_frame = cv2.resize(display_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Preprocessed View', display_frame)
            
            # Get model prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(processed_frame).unsqueeze(0).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # Perform action
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Use same reward system as in main.py
            if terminated:
                reward = -1.0
            elif info.get('score', 0) > 0:
                reward = 1.0
            else:
                reward = 0.1
                
            total_reward += reward
            
            # Break if 'q' is pressed
            if cv2.waitKey(20) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return
            
            if terminated or truncated:
                print(f"Game Over! Total Reward: {total_reward}")
                break

if __name__ == "__main__":
    run_model()
