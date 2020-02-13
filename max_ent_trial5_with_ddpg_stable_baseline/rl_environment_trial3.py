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
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.limit_values = np.array([1, 1])
        self.action_space_dim = 2
        self.dt = 0.1  # seconds between state updates
        self.t = 0
        self.t_limit = 540
        self.target = np.array([5.0, 5.0]).reshape((1, 2))
        self.threshold = 0.01
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-self.limit_values, self.limit_values)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # Valid action
        action = np.clip(action, -self.limit_values, self.limit_values)

        state = self.state
        x_val, y_val = state

        x_update, y_update = np.array([x_val, y_val]) + action
        # print("current state and action taken ", x_val, y_val, action, " result ", x_update, y_update)
        self.state = (x_update, y_update)

        goal = self.target
        done = False
        # print("x update and goal is ", x_update, goal[0][1], self.threshold)
        if x_update - goal[0][0] < self.threshold and y_update - goal[0][1] < self.threshold:
            done = True

        self.t += 1

        if self.t >= self.t_limit:
            done = True
        # print("goal shape is ", goal.shape)
        reward = self.compute_reward(np.array([self.state]), goal)
        obs = np.array([x_update, y_update])
        x_val, y_val = x_update, y_update
        return obs, reward, done, {}

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -d

    def _reset(self):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array([0.0, 0.0]),
                                           scale=np.array([0.02, 0.02]))
        self.steps_beyond_done = None
        self.t = 0  # timestep
        x_val, y_val = self.state
        obs = np.array([x_val, y_val])
        return obs

    def _render(self, mode='human', close=False):
        pass

