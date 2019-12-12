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
        self.lin_space_limits = np.linspace(0, 1, self.grid_size, dtype='float32')
        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        self.possible_actions = [i for i in range(27)]
        # Total Number of states defining the state of the robot
        self.n_states = 3
        self.rewards = []
        self.features = []
        self.state_action_value = []
        self.discount = 0
        self.current_pos = 1000
        self.terminal_state_val = 10
        self.weights = weights

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
            dict = {i: state_set[i]}
            self.states.update(dict)

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
            dict = {i: action_set[i]}
            self.action_space.update(dict)

        return self.action_space

    def get_state_val_index(self, state_val):
        index_val = int(state_val[0]) * pow(self.grid_size, 2) + int(state_val[1]) * pow(self.grid_size, 1) + \
                    int(state_val[2])
        return index_val*(self.grid_size-1)

    def is_terminal_state(self, state):

        # because terminal state is being given in index val
        if state == self.states[self.terminal_state_val]:
        # If terminal state is being given as a list then if state == self.terminal_state_val:
            return 1
        else:
            # It has not yet reached the terminal state
            return 0

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

    def reset(self):
        self.current_pos = np.random.randint(0, len(self.states))
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        return self.current_pos

    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_states):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i]))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value, features value and number of features based on the features provided
        reward, features_arr, len_features = self.reward_func(resulting_state[0], resulting_state[1], resulting_state[2], self.weights)
        # Checks if the resulting state is moving it out of the grid
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            return resulting_state, reward, self.is_terminal_state(resulting_state), None
        else:
            # If movement is out of the grid then just return the current state value itself
            print "*****The value is moving out of the grid in step function*******"
            return curr_state, reward, \
                   self.is_terminal_state(self.states[curr_state]), None


    def action_space_sample(self):
        print "random action choice ", np.random.choice(self.action_space)
        return np.random.choice(self.action_space)

def max_action(Q, state_values, action_values):
    print "max action state val ", state_values
    values = np.array([Q[state_values, a] for a in action_values])
    print "values in max action is ", values
    action = np.argmax(values)
    return action_values[action]

def q_learning(env_obj, alpha, gamma, epsilon):

    Q = {}
    # print "obj state ", env_obj.states.keys()
    # print "obj action ", env_obj.action_space.keys()

    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env_obj.reset()
        while not done:
            rand = np.random.random()
            # print "random val is ", rand
            # print "state val inside loop ", observation
             #print "action val inside loop", env_obj.action_space.keys()
            action = max_action(Q, observation, env_obj.action_space.keys()) if rand < (1 - epsilon) \
                else env_obj.action_space_sample
            observation_, reward, done, info = env_obj.step(observation, action)
            epRewards += reward
            next_observation_index = env_obj.get_state_val_index(observation_)
            action_ = max_action(Q, next_observation_index, env_obj.action_space.keys())
            print "next action val is ", action_
            Q[observation, action] = Q[observation, action] + alpha * (reward + \
                                                                       gamma * Q[next_observation_index, action_] -
                                                                       Q[observation, action])
            observation = next_observation_index
        if epsilon - 2 / numGames > 0:
            epsilon -= 2 / numGames
        else:
            epsilon = 0
        totalRewards[i] = epRewards
        return totalRewards


if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    obj_state_util = RobotStateUtils(11, weights)
    states = obj_state_util.create_state_space_model_func()
    # print states[100]
    action = obj_state_util.create_action_set_func()
    row_column = obj_state_util.get_state_val_index([0.0, 0.0, 1.0])
    # print "index val", row_column
    # print sorted(states.keys())
    # print sorted(states.values())
    state_check = 32
    action_val = 15
    # print "Current state index ", state_check
    r = obj_state_util.step(state_check, action_val)
    # print "r is ", r
    total_rewards = q_learning(obj_state_util, weights, gamma=0.09, epsilon=0.2)
    print "total rewards is ", total_rewards



    # obj_mdp = RobotMarkovModel()
    # rand_weights = np.random.rand(1, 3)
    # print rand_weights.shape
    # weights = np.array([[1, 1, 0]])
    # print weights.shape
    # reward, features, n_features = obj_mdp.reward_func(r[0], r[1], r[2], weights)
    # print reward
    # print features
    #
    # reward_trial = np.ones(len(states))
    # valuefunc = obj_state_util.calc_value_func(reward_trial, 0.01, 1e-2)
    # print "value function is ", valuefunc





















    # def get_model_indices(self, state_val):
    #     # Since everything is saved in a linear flattened form
    #     # Provides the z, y, x value of the current position based on the integer location value provided
    #     z = round(state_val % self.grid_size)
    #     y = round((state_val / self.grid_size) % self.grid_size)
    #     # x = round((self.current_pos-(self.current_pos // self.grid_size)) % self.grid_size)
    #     x = round((state_val / (self.grid_size * self.grid_size)) % self.grid_size)
    #     # Returns the actual value by dividing it by 10 (which is the scale of integer position and state values)
    #
    #     return [x/float(10), y/float(10), z/float(10)]

    '''
    def neighbouring(self, i, k):

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) + abs(i[2] - k[2]) <= 1

    def transition_probability(self, i, j, k):

        si, sj, sk = self.states[i]
        ai, aj, ak = self.action_space[j]
        s_ni, s_nj, s_nk = self.states[k]

        if not self.neighbouring((si, sj, sk), (s_ni, s_nj, s_nk)):
            return 0.0

        # Is k the intended state to move to?
        if (si + ai, sj + aj, sk + ak) == (s_ni, s_nj, s_nk):
            return 1

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (si, sj, sk) in {(0, 0, 0), (self.grid_size-1, self.grid_size-1, self.grid_size-1),
                            (0, self.grid_size-1, 0), (self.grid_size-1, 0, 0), (0, 0, self.grid_size-1)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= si + ai < self.grid_size and
                    0 <= sj + aj < self.grid_size and
                    0 <= sk + ak < self.grid_size):

                return 1
        else:
            # Not a corner. Is it an edge?
            if (si not in {0, self.grid_size-1} and
                sj not in {0, self.grid_size-1} and
                sk not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= si + ai < self.grid_size and
                    0 <= sj + aj < self.grid_size and
                    0 <= sk + ak < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1
            else:
                # We can blow off the grid only by wind.
                return 0

    def calc_value_func(self, reward, discount, threshold):
        n_states = len(self.states)
        n_actions = len(self.action_space)
        v = np.zeros(n_states)

        diff = float("inf")
        while diff > threshold:
            diff = 0
            for s in range(n_states):
                print "states is", s
                max_v = float("-inf")
                for a in range(n_actions):

                    # print "action is ", a
                     #print "max v is ", max_v
                    # print "reward + vdiscount ", tp.flatten().shape, reward.shape, v.shape, discount
                    max_v = max(max_v, np.dot(self.transition_probability(s, a, s1), reward + discount * v) for s1 in range(n_states))

                new_diff = abs(v[s] - max_v)
                if new_diff > diff:
                    diff = new_diff
                v[s] = max_v

        return v

    def find_policy(self, n_states, n_actions, reward, discount,
                    threshold=1e-2, v=None):

        if v is None:
            v = self.calc_value_func(reward, discount, threshold)
        def policy(s):
            return max(range(n_actions),
                       key=lambda a: sum((reward[k] + discount * v[k]) for k in range(n_states)))

        policy = np.array([policy(s) for s in range(n_states)])
        return policy
    '''
    '''
    def generate_trajectories(self, n_trajectories, trajectory_length, policy, random_start=True):

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy, sz = rn.randint(self.grid_size), rn.randint(self.grid_size), rn.randint(self.grid_size)
                print "Randomly assigned states are ", sx, sy, sz

            else:
                sx, sy, sz = 0, 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                # Follow the given policy.
                print "states are ", sx, sy, sz
                print "Policy ", self.get_state_val_index([sx, sy, sz])
                print "Ended "
                action = self.actions[policy(self.get_state_val_index([sx, sy, sz]))]

                if (0 <= sx + action[0] < self.grid_size and 0 <= sy + action[1] < self.grid_size
                        and 0 <= sz + action[2] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                    next_sz = sz + action[2]
                else:
                    next_sx = sx
                    next_sy = sy
                    next_sz = sz

                state_int = self.point_to_int((sx, sy, sz))
                # print "action is ", action
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy, next_sz))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy
                sz = next_sz

            trajectories.append(trajectory)

        return np.array(trajectories)
    '''


