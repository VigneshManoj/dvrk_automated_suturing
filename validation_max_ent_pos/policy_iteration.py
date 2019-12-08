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
        for pos_x in [-0.001, 0, 0.001]:
            for pos_y in [-0.001, 0, 0.001]:
                for pos_z in [-0.001, 0, 0.001]:
                    self.action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(self.action_set)):
            self.action_space[i] = self.action_set[i]

        return self.action_space

    def getRowAndColumn(self):
        # Since everything is saved in a linear flattened form
        # Provides the z, y, x value of the current position based on the integer location value provided
        z = round(self.current_pos % self.grid_size)
        y = round((self.current_pos / self.grid_size) % self.grid_size)
        # x = round((self.current_pos-(self.current_pos // self.grid_size)) % self.grid_size)
        x = round((self.current_pos / (self.grid_size * self.grid_size)) % self.grid_size)
        # Returns the actual value by dividing it by 10 (which is the scale of integer position and state values)

        return [x/float(10), y/float(10), z/float(10)]

if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    obj = RobotStateUtils(11)
    ele = obj.create_state_space_model_func()
    # print ele
    states = obj.create_state_space_model_func()
    print states[1000]
    # print len(states)
    action = obj.create_action_set_func()
    # print len(action)
    # print len(obj.possible_actions)
    row_column = obj.getRowAndColumn()
    print row_column