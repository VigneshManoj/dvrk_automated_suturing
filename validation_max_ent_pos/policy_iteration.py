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
        # Used for temporary assignment of action and state values while
        self.action_set = []
        self.state_set = []
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
        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    self.state_set.append([round(i_val, 1), round(j_val, 1), round(k_val, 1)])
        # Assigning the dictionary keys
        for i in range(len(self.state_set)):
            self.states[i] = self.state_set[i]
        return self.states

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        for pos_x in [-0.1, 0, 0.1]:
            for pos_y in [-0.1, 0, 0.1]:
                for pos_z in [-0.1, 0, 0.1]:
                    self.action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(self.action_set)):
            self.action_space[i] = self.action_set[i]

        return self.action_space

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


    def get_model_indices(self, state_val):
        x = int(state_val[0]) * pow(self.grid_size, 2) + int(state_val[1]) * pow(self.grid_size, 1) + int(state_val[2])
        return x*10

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
        print xyz_list
        print self.action_space[action]
        for i in range(0, len(xyz_list)):
            resulting_state.append(xyz_list[i] + self.action_space[action][i])

        return resulting_state
    '''
        reward = self.reward_func(curr_state, resulting_state)
        if not self.off_grid_move(resulting_state, self.current_pos):
            self.set_state(resulting_state)
            return resulting_state, reward, None
        else:
            return self.current_pos, reward, None

    def reward_func(self, curr_state, resulting_state):
        next_x, next_y, next_z = self.get_model_indices(resulting_state)
        x, y, z = self.get_model_indices(resulting_state)

    '''

if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    obj = RobotStateUtils(11)
    ele = obj.create_state_space_model_func()
    # print ele
    states = obj.create_state_space_model_func()
    # print states[33]
    action = obj.create_action_set_func()
    row_column = obj.get_model_indices([1.0, 0, 1.0])
    print row_column
    print states[1220]
    state = 32
    action_val = 1
    next_state = obj.step(state, action_val)
    print next_state

