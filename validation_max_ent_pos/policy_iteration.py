import numpy as np
import numba as nb
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size):
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

    def off_grid_move(self, new_state, old_state):
        # if we move into a row not in the grid
        if new_state not in self.states:
            return True
        # if we're trying to wrap around to next row
        elif old_state % self.grid_size == 0 and new_state % self.grid_size == self.grid_size - 1:
            return True
        elif old_state % self.grid_size == self.grid_size - 1 and new_state % self.grid_size == 0:
            return True
        else:
            return False

    def reset(self):
        self.current_pos = np.random.randint(0, len(self.states))
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        return self.current_pos

    def step(self, curr_state, action):
        resulting_state = []
        print "current state", self.states[curr_state]
        print "action taken", self.action_space[action]

        for i in range(0, self.n_states):
            resulting_state.append(self.states[curr_state][i] + self.action_space[action][i])

        return resulting_state

    def neighbouring(self, i, k):

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) + abs(i[2] - k[2]) <= 1

    def _transition_probability(self, i, j, k):

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
                    tp = np.dstack([a]*self.grid_size)
                    # print "action is ", a
                     #print "max v is ", max_v
                    # print "reward + vdiscount ", tp.flatten().shape, reward.shape, v.shape, discount
                    max_v = max(max_v, np.dot(tp, reward + discount * v))

                new_diff = abs(v[s] - max_v)
                if new_diff > diff:
                    diff = new_diff
                v[s] = max_v

        return v
    '''
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


if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    obj_state_util = RobotStateUtils(11)
    states = obj_state_util.create_state_space_model_func()
    print len(states)
    action = obj_state_util.create_action_set_func()
    row_column = obj_state_util.get_state_val_index([-0.1, 0.1, 1.0])
    # print row_column
    # print sorted(states.keys())
    # print sorted(states.values())
    state_check = 32
    action_val = 1
    # print "Current state index ", state_check
    r = obj_state_util.step(state_check, action_val)
    # print r
    obj_mdp = RobotMarkovModel()
    # rand_weights = np.random.rand(1, 3)
    # print rand_weights.shape
    weights = np.array([[1, 1, 0]])
    # print weights.shape
    reward, features, n_features = obj_mdp.reward_func(r[0], r[1], r[2], weights)
    # print reward
    # print features

    reward_trial = np.ones(len(states))
    valuefunc = obj_state_util.calc_value_func(reward_trial, 0.01, 1e-2)
    print "value function is ", valuefunc





















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


