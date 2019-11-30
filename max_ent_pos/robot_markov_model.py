import numpy as np
import numba as nb
import math


class RobotMarkovModel:
    def __init__(self):
        # Commented all this out currently because it was separately created in another function
        # since it was not being used here
        '''
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
        '''
        # Reads the trajectory data from the file
        trajectories = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data_without_norm.csv",
             delimiter=",")
        # Separates the state trajectories data and action data
        self.state_trajectories = trajectories[:, 3:6]
        self.action_trajectories = trajectories[:, 9:12]
        # Initialize the actions possible
        self.action_set = []
    '''
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
    '''
    # Returns the state and action array of expert trajectory
    def trajectories_data(self):
        # Return trajectories data if any function requires it outside this class
        return self.state_trajectories, self.action_trajectories

    # Returns the trajetory and rewards of the expert trajectory data
    def trajectories_features_rewards_array(self, weights):
        # Creates the array of features and rewards for the whole trajectory
        # It calls the RobotMarkovModel class reward function which returns the reward and features for that specific
        # state values. These values are repeatedly added until the length of trajectory
        trajectories_reward = []
        trajectories_features = []
        trajectory_reward = np.zeros([1, 1], dtype='float32')
        trajectory_features = np.zeros([2, 1], dtype='float32')
        for i in range(0, self.state_trajectories.shape[0]):
            # Reads only the state trajectory data and assigns the variables value of the first set of state values
            end_pos_x = self.state_trajectories[i, 0]
            end_pos_y = self.state_trajectories[i, 1]
            end_pos_z = self.state_trajectories[i, 2]
            # Calls the rewards function which returns the reward and features for that specific set of state values
            rewards, features = self.reward_func(end_pos_x, end_pos_y, end_pos_z, weights)
            # Sum up all the rewards of the trajectory
            trajectory_reward = trajectory_reward + rewards
            # Sum up all the rewards of the features
            trajectory_features = trajectory_features + np.vstack((features[0], features[1]))
        # Create a list of all the trajectory rewards and features
        trajectories_reward.append(trajectory_reward)
        trajectories_features.append(trajectory_features)
        # Returns the array of trajectory features and reward
        return trajectories_features, trajectories_reward

    # Calculates reward function
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered
        features = [self.features_array_prim_func, self.features_array_sec_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
            # print "alpha size", alpha[0, n].shape
            # print "features size ", features_arr[n].shape
            reward = reward + alpha[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return reward, features_arr

    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_prim_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value divided by
    # the variance value
    def features_array_sec_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2)
        # print f2
        return feature_2

    # It returns the features stacked together for a specific states (depends on how many number of features exist)
    def features_func(self, end_pos_x, end_pos_y, end_pos_z):

        features = [self.features_array_prim_func, self.features_array_sec_func]
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return features_arr

    # Returns the list of all the features and sum of features of an expert trajectory as an Numpy array
    def trajectories_features_array(self):
        # Creates the array of features and rewards for the whole trajectory
        # It calls the RobotMarkovModel class reward function which returns the reward and features for that specific
        # state values. These values are repeatedly added until the length of trajectory
        complete_feature_array = []
        features = 0
        sum_trajectories_features = []
        trajectory_features = np.zeros([2, 1], dtype='float32')
        for i in range(0, self.state_trajectories.shape[0]):
            # Reads only the state trajectory data and assigns the variables value of the first set of state values
            end_pos_x = self.state_trajectories[i, 0]
            end_pos_y = self.state_trajectories[i, 1]
            end_pos_z = self.state_trajectories[i, 2]

            # Calls the rewards function which returns features for that specific set of state values
            features = self.features_func(end_pos_x, end_pos_y, end_pos_z)
            # Creates a list of all the features
            complete_feature_array.append(features)
            trajectory_features = trajectory_features + np.vstack((features[0], features[1]))
        # Calculates the sum of all the trajectory feature values
        sum_trajectories_features.append(trajectory_features)
        # Returns the array of trajectory features and returns the array of all the features
        return np.array(sum_trajectories_features), np.array(complete_feature_array)

