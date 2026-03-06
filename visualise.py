import gymnasium as gym
import numpy as np

K = 5

# Best keyframes found by EA
keyframes = [
    np.array([-0.7030,  0.7973, -0.3660, -0.5685,  0.3985, -0.1988, 10]),
    np.array([ 0.1383,  0.0233,  0.0941,  0.1907,  0.3521,  0.1629, 27]),
    np.array([ 0.2776, -0.2377,  0.1615,  0.2369, -0.2976, -0.4349, 10]),
    np.array([-1.0000, -0.9888, -0.1711,  0.2570,  0.2395,  0.1347, 10]),
    np.array([ 0.9085, -0.3813, -0.2339,  0.1104,  0.6053,  0.7700, 10]),
]

env = gym.make("HalfCheetah-v5", render_mode="human")

for ep in range(3):
    obs, info = env.reset()  
    ki = 0
    remaining = int(keyframes[ki][-1])
    ep_reward = 0.0
    done = False

    while not done:
        action = keyframes[ki][:-1]

        obs, reward, term, trunc, info = env.step(action)
        ep_reward += reward

        env.render()

        remaining -= 1
        if remaining == 0:
            ki = (ki + 1) % K
            remaining = int(keyframes[ki][-1])

        done = term or trunc

    print(f"Episode {ep+1}: reward = {ep_reward:.2f}")

env.close()