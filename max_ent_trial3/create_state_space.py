import numpy as np
import numba as nb
import math


class RobotMarkovModel:
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
        trajectories = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data_without_norm.csv",
             delimiter=",")
        self.state_trajectories = trajectories[:, 0:6]
        self.action_trajectories = trajectories[:, 6:12]

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
        # Return trajectories data if any function requires it outside this class
        return self.state_trajectories, self.action_trajectories

    def trajectories_features_array(self, weights):
        # Creates the array of features and rewards for the whole trajectory
        # It calls the RobotMarkovModel class reward function which returns the reward and features for that specific
        # state values. These values are repeatedly added until the length of trajectory
        trajectories_reward = []
        trajectories_features = []
        trajectory_reward = np.zeros([1, 1], dtype='float32')
        trajectory_features = np.zeros([2, 1], dtype='float32')
        for i in range(0, self.state_trajectories.shape[0]):
            # Reads only the state trajectory data
            rot_par_r = self.state_trajectories[i, 0]
            rot_par_p = self.state_trajectories[i, 1]
            rot_par_y = self.state_trajectories[i, 2]
            end_pos_x = self.state_trajectories[i, 3]
            end_pos_y = self.state_trajectories[i, 4]
            end_pos_z = self.state_trajectories[i, 5]

            rewards, features = self.reward_func(rot_par_r, rot_par_p, rot_par_y,
                                                             end_pos_x, end_pos_y, end_pos_z, weights)
            trajectory_reward = trajectory_reward + rewards
            trajectory_features = trajectory_features + np.vstack((features[0], features[1]))
        trajectories_reward.append(trajectory_reward)
        trajectories_features.append(trajectory_features)
        # Returns the array of trajectory features and reward
        return trajectories_features, trajectories_reward

    # Calculates reward function
    def reward_func(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z, weights):

        features = [self.features_array_func, self.features_array_sec_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z))
            reward = reward + weights[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return reward, features_arr

    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_func(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value divided by
    # the variance value
    def features_array_sec_func(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2)
        # print f2
        return feature_2
