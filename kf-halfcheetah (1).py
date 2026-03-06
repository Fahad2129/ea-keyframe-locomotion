import gymnasium as gym
import numpy as np

# Initialise the environment
env = gym.make("HalfCheetah-v5", render_mode="human")

# Define keyframes with associated timestep durations
K = 5  # Number of keyframes in the cycle
keyframes = [
    np.array([-0.10,  0.05, -0.08,  0.12, -0.06,  0.03, 25]),
    np.array([ 0.20, -0.15,  0.30, -0.10,  0.25, -0.05, 40]),
    np.array([ 0.80, -0.40,  0.60, -0.30,  0.50, -0.20, 15]),
    np.array([ 0.50, -0.20,  0.40, -0.15,  0.30, -0.10, 20]),
    np.array([ 0.10,  0.00, -0.10,  0.05, -0.05,  0.02, 30])
]

# Reset the environment
observation, info = env.reset(seed=42)

keyframe_index = 0
remaining_steps = int(keyframes[keyframe_index][-1])

episode_reward = 0

while True:
    # Select action from the current keyframe
    action = keyframes[keyframe_index][:-1]

    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    # Render the environment
    env.render()
    
    # Decrement the remaining steps for this keyframe
    remaining_steps -= 1

    # If the keyframe duration is completed, move to the next keyframe
    if remaining_steps == 0:
        keyframe_index = (keyframe_index + 1) % K  # Cycle through keyframes
        remaining_steps = int(keyframes[keyframe_index][-1])

    # If the episode has ended (time limit reached)
    if terminated or truncated:
        print(f"Episode ended. Total reward: {episode_reward}")
        episode_reward = 0
        observation, info = env.reset()
        keyframe_index = 0
        remaining_steps = int(keyframes[keyframe_index][-1])

# Close the environment (good practice)
# env.close()