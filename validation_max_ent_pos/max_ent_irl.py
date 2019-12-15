import numpy as np
import numpy.random as rn
import math
from itertools import product
from policy_iteration import q_learning


# class MaxEntIRL(RobotStateUtils):
class MaxEntIRL():

    def __init__(self, trajectory_length):
        # super(MaxEntIRL, self).__init__(11)
        self.trajectory_length = trajectory_length

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, sum_trajectory_features_array, complete_features_array, discount,
                             n_trajectories, epochs, learning_rate, n_policy_iter, weights):
        # Finds the total number of states and dimensions of the list of features array
        n_states, d_states = complete_features_array.shape
        # Default value for now
        n_features = 3
        # print "complete feature", complete_features_array
        # print "length of action set", len(action_set)
        # Initialize alpha with random weights based on the dimensionality of the feature space
        # alpha = rn.uniform(size=(d_states,))
        # print "alpha is ", alpha
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(sum_trajectory_features_array, n_states)
        # Gradient descent on alpha
        for i in range(epochs):
            # print("i: {}".format(i))
            # Multiplies the features with randomized alpha value, size of output Ex: dot(449*2, 2x1)
            # Not required: self.reward = complete_features_array.dot(alpha)
            # expected_svf = self.find_expected_svf(complete_features_array, discount, n_trajectories,
            #                                                           n_policy_iter, weights, n_states)
            optimal_policy, state_features, expected_svf = self.find_expected_svf(weights, discount)
            # print "shape of features and svf is ", expected_svf
            # print "features is ", state_space_model_features
            # grad = feature_expectations - state_space_model_features.dot(expected_svf)
            # print "---shapes ----- \n", feature_expectations.reshape(2, 1).shape, expected_svf.reshape(2, 1).shape
            # print "Complete features array shape ", complete_features_array.shape
            # print "expected svf shape ", expected_svf.shape
            grad = feature_expectations.reshape(n_features, 1) - (state_features*expected_svf).reshape(n_features, 1)

            weights += learning_rate * np.transpose(grad)
            print "weights is ", weights

        return complete_features_array.dot(weights.reshape(n_features, 1)), weights

        # return self.reward

    def find_feature_expectations(self, trajectory_features, n_states):
        # Takes the sum of all the expert features as input
        feature_expectations = trajectory_features
        # Divide by the number of trajectories data available
        feature_expectations /= n_states
        # Return the expert data feature expectations
        return feature_expectations

    def find_expected_svf(self, weights, discount):
        optimal_policy, state_features, expected_svf = q_learning(weights=weights, alpha=0.1,
                                                                  gamma=discount, epsilon=0.2)

        return optimal_policy, state_features, expected_svf




        # robot_state_utils = RobotStateUtils()
        # rewards, state_space_model_features, n_features = robot_state_utils.calculate_optimal_policy_func(alpha, discount)
        # # policy = find_policy(n_states, r, n_actions, discount, transition_probability)
        # policy = find_policy(n_states, 8, rewards, discount)
        # # print "state space model features ", state_space_model_features
        # model_state_val_x, model_state_val_y, model_state_val_z, index_val_x, index_val_y, index_val_z = robot_state_utils.return_model_state_values()
        # mu = np.exp(-model_state_val_x ** 2) * np.exp(-model_state_val_y ** 2) * np.exp(-model_state_val_z ** 2)
        # action_set = robot_state_utils.return_action_set()
        # mu_reshape = np.reshape(mu, [11 * 11 * 11, 1])
        # mu = mu / sum(mu_reshape)
        # mu_last = mu
        # # print "Initial State Frequency calculated..."
        # for time in range(0, self.trajectory_length):
        #     s = np.zeros([11, 11, 11])
        #     for act_index, action in enumerate(action_set):
        #         new_state_val_x, new_state_val_y, new_state_val_z = robot_state_utils.get_next_state(model_state_val_x, model_state_val_y, model_state_val_z, action)
        #
        #         new_index_val_x, new_index_val_y, new_index_val_z = robot_state_utils.get_indices(new_state_val_x, new_state_val_y, new_state_val_z)
        #
        #         p = policy[act_index, index_val_x, index_val_y, index_val_z]
        #         s = s + p * mu_last[new_index_val_x, new_index_val_y, new_index_val_z]
        #     mu_last = s
        #     mu = mu + mu_last
        # mu = mu / self.trajectory_length
        # # mu = mu / n_time
        # state_visitation = mu_last * state_space_model_features
        # # print "State Visitation Frequency calculated."
        # return np.sum(state_visitation.reshape(n_features, 11 * 11 * 11), axis=1), policy, n_features
        # # return mu_last, policy, state_space_model_features
        # # state_visitation = mu_last * self.f
        # # print "State Visitation Frequency calculated."
        # # return np.sum(state_visitation.reshape(2, 11 * 11 * 11), axis=1), policy







