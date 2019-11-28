import numpy as np
import numba as nb
import math


class MaxEntIRL:

    def max_ent_irl(self, traj_features_array, action_set, discount, state_traj_array,
                         action_traj_array, epochs, learning_rate):

        feature_expectations = self.find_feature_expectations(traj_features_array)
        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            r = feature_matrix.dot(alpha)
            expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                             transition_probability, trajectories)
            grad = feature_expectations - feature_matrix.T.dot(expected_svf)

            alpha += learning_rate * grad

        return feature_matrix.dot(alpha).reshape((n_states,))

    def find_feature_expectations(self, traj_features):
        # Takes the sum of all the expert features as input
        feature_expectations = traj_features
        # Divide by the number of trajectories data available
        feature_expectations /= traj_features.shape[0]
        # Return the expert data feature expectations
        return feature_expectations



