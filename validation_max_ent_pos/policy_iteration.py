import numpy as np
import numba as nb
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_angle = [[-0.5, 0.5], [-0.234, -0.155], [0.28, 0.443]]
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 6 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.lin_space_limits = np.linspace(0, 1, 11, dtype='float32')
        self.states = {}
        self.action_space = {}
        self.action_set = []
        self.state_set = []
        # There are 6 parameters defining a state value of the robot, RPY and XYZ
        self.n_states = 3
        # The created model state values
        self.model_end_pos_x = []
        self.model_end_pos_y = []
        self.model_end_pos_z = []
        # The created model values index positions
        self.model_index_pos_x = []
        self.model_index_pos_y = []
        self.model_index_pos_z = []
        # The next state values index positions and position values of the new state
        self.model_new_index_pos_x = []
        self.model_new_index_pos_y = []
        self.model_new_index_pos_z = []
        self.model_new_end_pos_x = []
        self.model_new_end_pos_y = []
        self.model_new_end_pos_z = []
        self.rewards = []
        self.features = []
        self.state_action_value = []
        self.discount = 0

    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        # print "Creating State space "

        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    self.state_set.append([i_val, j_val, k_val])

        for i in range(len(self.state_set)):
            self.states[i] = self.state_set[i]
        return self.states

    # def create_action_space(self):
    #     values = [-0.001, 0, 0.001]
    #     keys = range(28)
    #     for i in keys:
    #         self.actionSpace[i] = values[i]
    #     return self.actionSpace

    def create_action_set_func(self):

        # Creates the action space required for the robot. It is defined by the user beforehand itself
        for pos_x in [-0.001, 0, 0.001]:
            for pos_y in [-0.001, 0, 0.001]:
                for pos_z in [-0.001, 0, 0.001]:
                    self.action_set.append([pos_x, pos_y, pos_z])
        for i in range(len(self.action_set)):
            self.action_space[i] = self.action_set[i]

        return self.action_space


if __name__ == '__main__':
    obj = RobotStateUtils()
    ele = obj.create_state_space_model_func()
    # print ele
    states = obj.create_state_space_model_func()
     #print states
    # print len(states)
    action = obj.create_action_set_func()
    # print action