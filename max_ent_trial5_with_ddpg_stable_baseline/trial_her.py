import gym
import gym_robot
import numpy as np

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise
from ambf_psm_env import AmbfPSMEnv

ENV_NAME = 'psm/pitchendlink'

# Number of actions for your environment
num_actions_input = 7
num_states_input = 7
num_goal_input = 6
# Get the environment and extract the number of actions.
env = AmbfPSMEnv(n_actions=num_actions_input, n_states=num_states_input, n_goals=num_goal_input)
env.make(ENV_NAME)
env.reset()

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# # SAC hyperparams:
# model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]))

# DDPG Hyperparams:
# NOTE: it works even without action noise
n_actions = env.action_space.shape[0]
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
model = HER('MlpPolicy', env, DDPG, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            actor_lr=1e-3, critic_lr=1e-3, action_noise=action_noise,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))


model.learn(int(2e5))
model.save('gym_robot')

# Load saved model
model = HER.load('gym_robot', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(100):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      env.render()
      episode_reward += reward
      if done or info.get('is_success', False):
              print("Reward:", episode_reward, "Success?", info.get('is_success', False))
              episode_reward = 0.0
              obs = env.reset()