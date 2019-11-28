import numpy as np
import numpy.random as rn
import math


class MaxEntIRL:

    def max_ent_irl(self, trajectory_features_array,  complete_features_array, action_set, discount,
                    state_trajectory_array, action_trajectory_array, epochs, learning_rate):
        n_states, d_states = complete_features_array.shape
        print "c feature", complete_features_array.shape
        alpha = rn.uniform(size=(d_states,))
        print "alpha is ", alpha
        feature_expectations = self.find_feature_expectations(trajectory_features_array)
        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            rewards = complete_features_array.dot(alpha)
            expected_svf = find_expected_svf(n_states, r, n_actions, discount, trajectories)
            grad = feature_expectations - complete_features_array.T.dot(expected_svf)
            #
            alpha += learning_rate * grad

        return complete_features_array.dot(alpha).reshape((n_states,))

        # return rewards

    def find_feature_expectations(self, trajectory_features):
        # Takes the sum of all the expert features as input
        feature_expectations = trajectory_features
        # Divide by the number of trajectories data available
        feature_expectations /= trajectory_features.shape[0]
        # Return the expert data feature expectations
        return feature_expectations






