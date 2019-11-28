import numpy as np
import numpy.random as rn
import math
from robot_state_utils import RobotStateUtils

class MaxEntIRL:
    def __init__(self, trajectory_length):
        self.reward = 0
        self.trajectory_length = trajectory_length

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, trajectory_features_array, complete_features_array, discount,
                             n_trajectories, epochs, learning_rate):
        # Finds the total number of states and dimesions of the list of features array
        n_states, d_states = complete_features_array.shape
        # print "c feature", complete_features_array.shape
        # print "length of action set", len(action_set)
        # Initialize alpha with random weights based on the dimensionality of the feature space
        alpha = rn.uniform(size=(d_states,))
        print "alpha is ", alpha
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(trajectory_features_array)
        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            # Multiplies the features with randomized alpha value, size of output Ex: dot(449*2, 2x1)
            self.reward = complete_features_array.dot(alpha)
            expected_svf = self.find_expected_svf(discount, n_trajectories)
            # grad = feature_expectations - complete_features_array.T.dot(expected_svf)
            #
            # alpha += learning_rate * grad

        # return complete_features_array.dot(alpha).reshape((n_states,))

        return self.reward

    def find_feature_expectations(self, trajectory_features):
        # Takes the sum of all the expert features as input
        feature_expectations = trajectory_features
        # Divide by the number of trajectories data available
        feature_expectations /= trajectory_features.shape[0]
        # Return the expert data feature expectations
        return feature_expectations

    def find_expected_svf(self, discount, n_trajectories):
        # Trajectory length is calculated as follows:
        # Trajectory is basically list of all Traj1 Traj2 Traj3 Traj4 of user 1 and
        # similarly  Traj1 Traj2 Traj3 Traj4 for user 2
        # So basically inside trajectory is the list of all states visited
        # Outisde trajectory value would be the different trajectories collected from user (say trial1 trial2 etc)
        # So in this case currently, it is 1 in our case
        robot_state_utils = RobotStateUtils()
        # policy = find_policy(n_states, r, n_actions, discount, transition_probability)
        policy = robot_state_utils.calculate_optimal_policy_func(self.reward, discount)

        for t in range(1, self.trajectory_length):
            expected_svf[:, t] = 0
            for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
                expected_svf[k, t] += (expected_svf[i, t - 1] *
                                       policy[i, j] *  # Stochastic policy
                                       transition_probability[i, j, k])

        return expected_svf.sum(axis=1)







