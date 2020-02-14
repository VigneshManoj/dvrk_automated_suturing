import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import gym_robot

env = gym.make('robot-v0')
i = 0
# the noise objects for DDPG
# print("actions space ", np.abs(env.action_space.low), np.abs(env.action_space.high))
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render_eval=False)
model.learn(total_timesteps=400000)
model.save("/home/vignesh/PycharmProjects/dvrk_automated_suturing/max_ent_trial5_with_ddpg_stable_baseline/ddpg_robot_1")

del model # remove to demonstrate saving and loading

model = DDPG.load("/home/vignesh/PycharmProjects/dvrk_automated_suturing/max_ent_trial5_with_ddpg_stable_baseline/ddpg_mountain")

print(env.action_space)
# while True:
#     obs = env.reset()
#
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     i += 1
#     if i % 1000 == 0:
#         print("current state, action is ", _states, action, " resulting state ", obs)
#     env.render()

for i in range(100):  # Total episodes to train
    obs = env.reset()
    print("observation ", obs)
    done = False
    while not done:
        action, _states = model.predict(obs)

        obs, rewards, done, info = env.step(action)
        # print("current state, action is ", _states, action, " resulting state ", obs)
        i += 1
        if i % 1000 == 0:
            print("current state, action is ", _states, action, " resulting state ", obs)
        # env.render('ansi')
