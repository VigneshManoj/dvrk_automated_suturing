import numpy as np


class RobotMarkovModel:
    def __init__(self):
        # Reads the trajectory data from the file
        trajectories1 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/check_data_max_ent_trial4_code1.csv",
             delimiter=",")
        trajectories2 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/check_data_max_ent_trial4_code2.csv",
             delimiter=",")
        trajectories3 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/check_data_max_ent_trial4_code3.csv",
             delimiter=",")
        # Separates the state trajectories data and action data
        self.state_trajectories = []
        self.state_trajectories.append(trajectories1[:, 0:3])
        self.state_trajectories.append(trajectories2[:, 0:3])
        self.state_trajectories.append(trajectories3[:, 0:3])

        self.action_trajectories = []
        self.action_trajectories.append(trajectories1[:, 3:6])
        self.action_trajectories.append(trajectories2[:, 3:6])
        self.action_trajectories.append(trajectories3[:, 3:6])

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
        individual_feature_array = []
        feature_array_all_trajectories = []
        sum_trajectory_features = np.zeros([3, 1], dtype='float32')
        for state_trajectory in self.state_trajectories:
            for i in range(0, len(state_trajectory)):
                # Reads only the state trajectory data and assigns the variables value of the first set of state values
                end_pos_x = state_trajectory[i, 0]
                end_pos_y = state_trajectory[i, 1]
                end_pos_z = state_trajectory[i, 2]

                # Calls the features function which returns features for that specific set of state values
                features = self.features_func(end_pos_x, end_pos_y, end_pos_z)
                # Creates a list of all the features
                individual_feature_array.append(features)
                sum_trajectory_features = sum_trajectory_features + np.vstack((features[0], features[1], features[2]))

            # Calculates the sum of all the trajectory feature values
            feature_array_all_trajectories.append(individual_feature_array)
        # Returns the array of sum of all trajectory features and returns the array of all the features of a trajectory
        return np.array(sum_trajectory_features), np.array(feature_array_all_trajectories)

if __name__ == '__main__':
    obj = RobotMarkovModel()
    sum_feat, feat_array = obj.generate_trajectories()
    print "sum of features ", sum_feat
    print "features array ", feat_array
    print "len ", len(feat_array)


