import numpy as np
import numba as nb
import math


class RobotMarkovModel:
    def __init__(self):
        # Commented all this out currently because it was separately created in another function
        # since it was not being used here
        # Reads the trajectory data from the file
        trajectories = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/check_data_max_ent_trial4_code.csv",
             delimiter=",")
        # Separates the state trajectories data and action data
        self.state_trajectories = trajectories[:, 0:3]
        self.action_trajectories = trajectories[:, 3:6]
        # Initialize the actions possible
        self.action_set = []
    # Returns the state and action array of expert trajectory
    def trajectories_data(self):
        # Return trajectories data if any function requires it outside this class
        return self.state_trajectories, self.action_trajectories

    # Calculates reward function
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered
        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
            # print "alpha size", alpha[0, n].shape
            # print "features size ", features_arr[n].shape
            reward = reward + alpha[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return reward, np.array([features_arr]), len(features)

    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_prim_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(end_pos_x**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value divided by
    # the variance value
    def features_array_sec_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(end_pos_y**2))
        # print f2
        return feature_2

    def features_array_tert_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_3 = np.exp(-(end_pos_z**2))
        return feature_3

    # It returns the features stacked together for a specific states (depends on how many number of features exist)
    def features_func(self, end_pos_x, end_pos_y, end_pos_z):

        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return features_arr


    def generate_trajectories(self):
        # Creates the array of features and rewards for the whole trajectory
        # It calls the RobotMarkovModel class reward function which returns the reward and features for that specific
        # state values. These values are repeatedly added until the length of trajectory
        complete_feature_array = []
        sum_trajectories_features = []
        trajectory_features = np.zeros([3, 1], dtype='float32')
        for i in range(0, self.state_trajectories.shape[0]):
            # Reads only the state trajectory data and assigns the variables value of the first set of state values
            end_pos_x = self.state_trajectories[i, 0]
            end_pos_y = self.state_trajectories[i, 1]
            end_pos_z = self.state_trajectories[i, 2]

            # Calls the rewards function which returns features for that specific set of state values
            features = self.features_func(end_pos_x, end_pos_y, end_pos_z)
            # Creates a list of all the features
            complete_feature_array.append(features)
            trajectory_features = trajectory_features + np.vstack((features[0], features[1], features[2]))
        # Calculates the sum of all the trajectory feature values
        sum_trajectories_features.append(trajectory_features)
        # Returns the array of trajectory features and returns the array of all the features
        return np.array(sum_trajectories_features), np.array(complete_feature_array)

if __name__ == '__main__':
    obj = RobotMarkovModel()
    sum_feat, feat_array = obj.generate_trajectories()
    print "sum of features ", sum_feat
    print "features array ", feat_array
    print "len ", len(feat_array)


