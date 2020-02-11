from ddpg_keras_suturing import Agent
# from plot_utils import plotLearning
import numpy as np
# import numba as nb
import math
import concurrent.futures
# from robot_markov_model import RobotMarkovModel
# import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, weights):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be

        self.action_space_low = -0.01
        self.action_space_high = 0.01
        # Numerical values assigned to each action in the dictionary
        # Total Number of states defining the state of the robot
        self.n_states = 2
        # self.current_pos = 1000
        self.terminal_state_val = np.array([0.1, 0.])
        self.weights = weights
        self.trans_prob = 1
    '''
    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 1), round(j_val, 1), round(k_val, 1)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states
    
    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        action_set = []
        for pos_x in [-0.1, 0, 0.1]:
            for pos_y in [-0.1, 0, 0.1]:
                for pos_z in [-0.1, 0, 0.1]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space
    
    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.5) * pow(self.grid_size, 2)) + abs((state_val[1] + 0.5) * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.5))
        return round(index_val*(self.grid_size-1))
    '''

    def distance_goal(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def reward_func(self, curr_state, goal_state):
        # Creates list of all the features being considered
        dist = self.distance_goal(curr_state, goal_state)
        if abs(dist) > 0.001:
            return -dist, False
        else:
            return -dist, True

    def reset(self):
        # self.pos = np.random.randint(0, len(self.states))
        self.pos = np.array([0, 0])
        return self.pos

    def step(self, curr_state, action):
        action = np.clip(action, self.action_space_low, self.action_space_high)
        resulting_state = np.zeros(self.n_states)
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_states):

            resulting_state[i] = curr_state[i] + action[i]
        print "values ", resulting_state, curr_state, action
        # Calculates the reward and returns the reward value, features value and
        reward, done = self.reward_func(resulting_state, self.terminal_state_val)
        # print "reward is ", reward
        return resulting_state, reward, done, None


def ddpg_model(env_obj, weights, alpha, gamma, epsilon):

    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=2, tau=0.001,
                  env=env_obj, batch_size=64, layer1_size=800, layer2_size=600,
                  n_actions=2)
    np.random.seed(0)
    score_history = []

    for i in range(1000):
        done = False
        score = 0
        observation = env_obj.reset()
        while not done:
            action = agent.choose_action(observation)
            # print "action chosen ", action
            observation_, reward, done, info = env_obj.step(observation, action)
            # print "resulting state is ", observation_
            agent.remember(observation, action, reward, observation_, int(done))
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))


if __name__ == '__main__':
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(weights)

    ddpg_model(env_obj, weights, alpha=0.01, gamma=0.99, epsilon=1.0)

    # x = [i+1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)
