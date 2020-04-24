"""
Cart pole swing-up: Identical version to PILCO V0.9
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

def goal_distance(goal_a, goal_b):
    # print("goal a and b shape ", goal_a.shape, goal_b.shape)
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class RobotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'ansi'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.limit_values = 1
        self.action_space_dim = 2
        self.dt = 0.1  # seconds between state updates

        self.target = np.array([5.0, 5.0])
        self.threshold = 0.01
        # high = np.array([
        #     np.finfo(np.float32).max,
        #     np.finfo(np.float32).max])

        self.action_space = spaces.Box(-1., 1., shape=(self.action_space_dim,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.action_space_dim,), dtype='float32')
        # self.action_space = spaces.Box(
        #     low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self._seed()
        self.viewer = None
        self.state = np.array([0., 0.])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print("action space is ", self.action_space)
        # Valid action
        action = np.clip(action, -1, +1).astype(np.float32)

        new_state = self.state + action
        print("current state and action taken ", self.state, action, " result and result shape ", new_state)
        self.state = new_state
        goal = self.target
        done = False
        # print("x update and goal is ", x_update, goal[0][1], self.threshold)
        if abs(new_state[0] - goal[0]) < self.threshold and abs(new_state[1] - goal[0]) < self.threshold:
            done = True

        # print("shape goal state action is ", goal.shape, np.array([x_val, y_val]).shape, action.shape)
        reward = self.compute_reward(self.state, goal)
        if abs(new_state[0]) >= 500.0 or abs(new_state[1]) >= 500.0:
            done = True
            reward -= 100
        print("reward received ", reward)
        return new_state, reward, done, {}

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -d

    def _reset(self):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        random_state = np.random.normal(loc=np.array([0.0, 0.0]), scale=np.array([0.02, 0.02]))
        obs = random_state
        # print("random state, random state shape, observation shape is ", random_state, random_state.shape, obs.shape)

        return obs

    def _render(self, mode='ansi', close=False):
        pass


# if __name__ == '__main__':
#     env = RobotEnv()
#     action = np.array([0.1, 0.1])
#     while True:
#         env.render()
#         obs, rewards, done, info = env.step(action)
#         # if i % 100 == 0:
#         # print("current state, action is ", _states, action, " resulting state ", obs)
#         env.render(mode='ansi')

