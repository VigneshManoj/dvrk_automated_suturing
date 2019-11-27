import numpy as np
import numba as nb
import math


class RobotStateSpace:
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
        # The created model state values
        self.model_rot_par_r = []
        self.model_rot_par_p = []
        self.model_rot_par_y = []
        self.model_end_pos_x = []
        self.model_end_pos_y = []
        self.model_end_pos_z = []

    def state_space_model(self):

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

    def trajectories_data(self):
        trajectories = np.genfromtxt(
            "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data_without_norm.csv",
            delimiter=",")
        state_trajectories = trajectories[:, 0:6]
        action_trajectories = trajectories[:, 6:12]

        return state_trajectories, action_trajectories
