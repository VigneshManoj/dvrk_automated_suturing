from dqn_keras_suturing import Agent
from plot_utils import plotLearning
import numpy as np
import numba as nb
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, weights):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.lin_space_limits = np.linspace(-0.5, 0.5, self.grid_size, dtype='float32')
        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        self.possible_actions = [i for i in range(27)]
        # Total Number of states defining the state of the robot
        self.n_states = 3
        # self.current_pos = 1000
        self.terminal_state_val = 18
        self.weights = weights
        self.trans_prob = 1

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

    def is_terminal_state(self, state):

        # because terminal state is being given in index val
        # x = self.states[self.terminal_state_val]
        if state == self.states[self.terminal_state_val]:
        # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
            # It has not yet reached the terminal state
            return False

    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        if new_state not in self.states.values():
            return True
        # if trying to wrap around the grid, also the reason for the for x in _ is because old_state is a list
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in new_state) == 0:
            return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False
    '''
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered
        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))

            reward = reward + alpha[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return reward, np.array([features_arr]), len(features)
    '''
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered

        # reward = -1
        if self.is_terminal_state([end_pos_x, end_pos_y, end_pos_z]):
            reward = 0
        else:
            reward = -1

        return reward, 1, 2

    '''
    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_prim_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(end_pos_x**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value
    def features_array_sec_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(end_pos_y**2))
        # print f2
        return feature_2

    # Created feature set3 which basically takes the exponential of sum of individually squared value
    def features_array_tert_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_3 = np.exp(-(end_pos_z**2))
        return feature_3

    def features_array_sum_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_4 = np.exp(-(end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
        return feature_4
    '''
    def reset(self):
        # self.pos = np.random.randint(0, len(self.states))
        self.pos = np.array([0, 0, 0])
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        return self.pos

    def step(self, curr_state, action):
        resulting_state = np.zeros(self.n_states)
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_states):
            resulting_state[i] = round(curr_state[i] + self.action_space[action][i], 1)

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value, features value and
        # number of features based on the features provided
        reward, features_arr, len_features = self.reward_func(resulting_state[0],
                                                              resulting_state[1],
                                                              resulting_state[2], self.weights)
        # print "reward is ", reward
        # Checks if the resulting state is moving it out of the grid
        # resulting_state_index = self.get_state_val_index(resulting_state)
        # curr_state_index = self.get_state_val_index(curr_state)

        if not self.off_grid_move(resulting_state.tolist(), curr_state.tolist()):
            return resulting_state, reward, self.is_terminal_state(resulting_state.tolist()), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return curr_state, reward, self.is_terminal_state(curr_state.tolist()), None
    '''
    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))

    def features_func(self, end_pos_x, end_pos_y, end_pos_z):

        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return features_arr

    def sum_of_features(self, state_trajectory):
        # Creates the array of features and rewards for the whole trajectory
        # It calls the RobotMarkovModel class reward function which returns the reward and features for that specific
        # state values. These values are repeatedly added until the length of trajectory
        n = len(state_trajectory)
        # print "n is ", n
        # print "state traj is ", state_trajectory
        # print "state traj is ", state_trajectory[0][0]
        sum_trajectories_features = []
        trajectory_features = np.zeros([3, 1], dtype='float32')
        for i in range(0, n):
            # Reads only the state trajectory data and assigns the variables value of the first set of state values
            end_pos_x = state_trajectory[i][0]
            end_pos_y = state_trajectory[i][1]
            end_pos_z = state_trajectory[i][2]

            # Calls the rewards function which returns features for that specific set of state values
            features = self.features_func(end_pos_x, end_pos_y, end_pos_z)
            # Creates a list of all the features
            trajectory_features = trajectory_features + np.vstack((features[0], features[1], features[2]))
        # Calculates the sum of all the trajectory feature values
        sum_trajectories_features.append(trajectory_features)
        # Returns the array of trajectory features and returns the array of all the features
        return np.array(sum_trajectories_features)
    '''
    def get_transition_states_and_probs(self, curr_state, action):

        if self.is_terminal_state(curr_state):
            return [(curr_state, 1)]
        resulting_state = []
        if self.trans_prob == 1:
            for i in range(0, self.n_states):
                resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))
            resulting_state_index = self.get_state_val_index(resulting_state)

            if not self.off_grid_move(resulting_state, self.states[curr_state]):
                # return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
                return [(resulting_state_index, 1)]
            else:
                # if the state is invalid, stay in the current state
                return [(curr_state, 1)]

    def get_transition_mat_deterministic(self):

        n_states = self.grid_size**3
        n_actions = len(self.action_space)
        P_a = np.zeros((n_states, n_actions), dtype=np.int32)
        for si in range(n_states):
            for a in range(n_actions):
                probs = self.get_transition_states_and_probs(si, a)

                for posj, prob in probs:
                    # Prob of si to sj given action a
                    prob = int(prob)
                    if prob == 1:
                        P_a[si, a] = posj
                    elif prob != 0:
                        raise ValueError('not a deterministic environment!')
        return P_a

    def compute_state_visition_freq(self, trajs, optimal_policy):
        P_a = self.get_transition_mat_deterministic()
        n_states, n_actions = P_a.shape
        # optimal_policy, trajs = q_learning(weights, alpha=0.1, gamma=0.9, epsilon=0.2)
        T = 2
        # print "T is ", T
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([n_states, T])
        # print "trajs is ", trajs
        mu[0, 0] = 1
        mu[:, 0] = mu[:, 0] / len(trajs)

        for s in range(n_states):
            for t in range(T-1):
                print "P_a is ", [P_a[pre_s, int(optimal_policy[pre_s])] for pre_s in range(n_states)]
                mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, int(optimal_policy[pre_s])] for pre_s in range(n_states)])
                print "mu val ", mu[s, t+1]
        p = np.sum(mu, 1)
        return p

'''
def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    action = np.argmax(q_values)
    # print "---max action function action ", action
    # print "max q value ", q_values[action]
    return action_values[action]
'''

def dqn_model(env_obj, weights, alpha, gamma, epsilon):
    lr = 0.0005
    load_checkpoint = False
    n_episodes = 500
    agent = Agent(gamma=gamma, epsilon=epsilon, alpha=alpha, input_dims=3,
                  n_actions=27, mem_size=1000000, batch_size=32, epsilon_end=0.0)
    if load_checkpoint:
        agent.load_model()
    scores = []
    eps_history = []

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env_obj.reset()
        while not done:
            action = agent.choose_action(observation)
            # print "action chosen ", action
            observation_, reward, done, info = env_obj.step(observation, action)
            # print "resulting state is ", observation_
            score += reward
            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()


if __name__ == '__main__':
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(10, weights)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    policy = np.zeros(len(states))
    print "states is ", states[0], states[18]
    print "actions are ", action

    dqn_model(env_obj, weights, alpha=0.01, gamma=0.99, epsilon=1.0)

    # x = [i+1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)
