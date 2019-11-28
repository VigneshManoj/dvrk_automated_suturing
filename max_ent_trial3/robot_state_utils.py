import numpy as np
import numba as nb
import math


class RobotStateUtils:
    def __init__(self):
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_angle = [[-0.5, 0.5], [-0.234, -0.155], [0.28, 0.443]]
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 6 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.model_limits_rot_r_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        self.model_limits_rot_p_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        self.model_limits_rot_y_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        self.model_limits_pos_x_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_limits_pos_y_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_limits_pos_z_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        # There are 6 parameters defining a state value of the robot, RPY and XYZ
        self.n_states = 6
        # The created model state values
        self.model_rot_par_r = []
        self.model_rot_par_p = []
        self.model_rot_par_y = []
        self.model_end_pos_x = []
        self.model_end_pos_y = []
        self.model_end_pos_z = []
        self.action_set = []

    def create_state_space_model_func(self):

        print "Creating State space "
        self.model_rot_par_r, self.model_rot_par_p, self.model_rot_par_y, self.model_end_pos_x, self.model_end_pos_y, \
        self.model_end_pos_z = np.meshgrid(self.model_limits_rot_r_val,
                                           self.model_limits_rot_p_val,
                                           self.model_limits_rot_y_val,
                                           self.model_limits_pos_x_val,
                                           self.model_limits_pos_y_val,
                                           self.model_limits_pos_z_val,
                                           sparse=True)
        print "State space has been created"

        return self.model_rot_par_r, self.model_rot_par_p, self.model_rot_par_y, \
               self.model_end_pos_x, self.model_end_pos_y, self.model_end_pos_z

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        for rot_r in [-0.01, 0, 0.01]:
            for rot_p in [-0.01, 0, 0.01]:
                for rot_y in [-0.01, 0, 0.01]:
                    for pos_x in [-0.001, 0, 0.001]:
                        for pos_y in [-0.001, 0, 0.001]:
                            for pos_z in [-0.001, 0, 0.001]:
                                self.action_set.append(np.array([rot_r, rot_p, rot_y, pos_x, pos_y, pos_z]))
        return self.action_set

    def calculate_optimal_policy_func(self, reward, discount):
        model_rot_par_r, model_rot_par_p, model_rot_par_y, \
        model_end_pos_x, model_end_pos_y, model_end_pos_z = self.create_state_space_model_func()
        action_set = self.create_action_set_func()
        n_actions = len(action_set)









