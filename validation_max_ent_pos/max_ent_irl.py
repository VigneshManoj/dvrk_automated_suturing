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
        print "complete feature", complete_features_array
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
            expected_svf = self.find_svf(n_states, complete_features_array)
            # print "shape of features and svf is ", expected_svf
            # print "features is ", state_space_model_features
            # grad = feature_expectations - state_space_model_features.dot(expected_svf)
            # print "---shapes ----- \n", feature_expectations.reshape(2, 1).shape, expected_svf.reshape(2, 1).shape
            print "Complete features array shape ", complete_features_array.shape
            print "expected svf shape ", expected_svf.shape
            grad = feature_expectations.reshape(n_features, 1) - complete_features_array.T.dot(expected_svf)
            #
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

    def find_expected_svf(self, trajectories, discount, n_trajectories, n_policy_iter, weights, n_states):
        # Trajectory length is calculated as follows:
        # Trajectory is basically list of all Traj1 Traj2 Traj3 Traj4 of user 1 and
        # similarly  Traj1 Traj2 Traj3 Traj4 for user 2
        # So basically inside trajectory is the list of all states visited
        # Outside trajectory value would be the different trajectories collected from user (say trial1 trial2 etc)
        # So in this case currently, it is 1 in our case

        # Write code to pass n_actions throughout the program to reach this function
        n_actions = 27
        trajectory_length = 1
        optimal_policy = q_learning(weights, alpha=0.1, gamma=0.9, epsilon=0.2)

        print "policy in maxent is ", optimal_policy
        start_state_count = np.zeros(n_states)
        p_start_state = 0.25
        # print "state state count ", start_state_count
        # print "n traj ", n_trajectories, trajectory_length
        # print "start state value ", p_start_state
        expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
        for t in range(1, trajectory_length):
            expected_svf[:, t] = 0
            for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
                expected_svf[k, t] += (expected_svf[i, t - 1] *
                                       optimal_policy[i, j])

        return np.sum(expected_svf)

    def find_svf(self, n_states, trajectories):
        svf = np.zeros(n_states)

        for state, _, _ in trajectories:
            svf[state] += 1

        svf /= trajectories.shape[0]
        # print "svf is ", svf
        return svf


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







