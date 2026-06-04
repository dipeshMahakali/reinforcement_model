import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the Environment
# CartPole-v1 gives rewards for keeping the pole upright.
env = gym.make("CartPole-v1")

# 2. Initialize the PPO Agent
# 'MlpPolicy' means we are using a standard Multi-Layer Perceptron neural network.
print("Initializing the PPO Agent...")
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the Agent
# 10,000 steps takes less than a minute on a standard computer.
print("Training started...")
model.learn(total_timesteps=10000)
print("Training completed!")

# 4. Evaluate the trained Agent
# This tests the agent over 10 consecutive episodes to get an average score.
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward} +/- {std_reward}")

# 5. Save the trained model parameters
model.save("ppo_cartpole_model")
print("Model saved as ppo_cartpole_model.zip")

# Always close the environment when done
env.close()
