import numpy as np
import numpy.random as rn
import math
from robot_state_utils import RobotStateUtils

class MaxEntIRL(RobotStateUtils):
    def __init__(self, trajectory_length):
        super(MaxEntIRL, self).__init__()
        self.trajectory_length = trajectory_length

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, trajectory_features_array, complete_features_array, discount,
                             n_trajectories, epochs, learning_rate, n_policy_iter, alpha):
        # Finds the total number of states and dimensions of the list of features array
        n_states, d_states = complete_features_array.shape
        print "complete feature", complete_features_array.shape
        # print "length of action set", len(action_set)
        # Initialize alpha with random weights based on the dimensionality of the feature space
        # alpha = rn.uniform(size=(d_states,))
        print "alpha is ", alpha
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(trajectory_features_array)
        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            # Multiplies the features with randomized alpha value, size of output Ex: dot(449*2, 2x1)
            # Not required: self.reward = complete_features_array.dot(alpha)
            expected_svf, policy = self.find_expected_svf(discount, n_trajectories, n_policy_iter, alpha)
            print "shape of features and svf is ", expected_svf.shape
            # print "features is ", state_space_model_features
            # grad = feature_expectations - state_space_model_features.dot(expected_svf)
            print "---shapes ----- \n", feature_expectations.reshape(2, 1).shape, expected_svf.reshape(2, 1).shape
            grad = feature_expectations.reshape(2, 1) - expected_svf.reshape(2, 1)
            #
            alpha += learning_rate * np.transpose(grad)

        return complete_features_array.dot(alpha.reshape(2, 1)), alpha

        # return self.reward

    def find_feature_expectations(self, trajectory_features):
        # Takes the sum of all the expert features as input
        feature_expectations = trajectory_features
        # Divide by the number of trajectories data available
        feature_expectations /= trajectory_features.shape[0]
        # Return the expert data feature expectations
        return feature_expectations

    def find_expected_svf(self, discount, n_trajectories, n_policy_iter, alpha):
        # Trajectory length is calculated as follows:
        # Trajectory is basically list of all Traj1 Traj2 Traj3 Traj4 of user 1 and
        # similarly  Traj1 Traj2 Traj3 Traj4 for user 2
        # So basically inside trajectory is the list of all states visited
        # Outisde trajectory value would be the different trajectories collected from user (say trial1 trial2 etc)
        # So in this case currently, it is 1 in our case
        robot_state_utils = RobotStateUtils()
        # policy = find_policy(n_states, r, n_actions, discount, transition_probability)
        policy, state_space_model_features = robot_state_utils.calculate_optimal_policy_func(alpha, discount, n_policy_iter)
        model_state_val_x, model_state_val_y, model_state_val_z, index_val_x, index_val_y, index_val_z = robot_state_utils.return_model_state_values()
        mu = np.exp(-model_state_val_x ** 2) * np.exp(-model_state_val_y ** 2) * np.exp(-model_state_val_z ** 2)
        action_set = robot_state_utils.return_action_set()
        mu_reshape = np.reshape(mu, [11 * 11 * 11, 1])
        mu = mu / sum(mu_reshape)
        mu_last = mu
        print "Initial State Frequency calculated..."
        for time in range(0, self.trajectory_length):
            s = np.zeros([11, 11, 11])
            for act_index, action in enumerate(action_set):
                new_state_val_x, new_state_val_y, new_state_val_z = robot_state_utils.get_next_state(model_state_val_x, model_state_val_y, model_state_val_z, action)

                new_index_val_x, new_index_val_y, new_index_val_z = robot_state_utils.get_indices(new_state_val_x, new_state_val_y, new_state_val_z)

                p = policy[act_index, index_val_x, index_val_y, index_val_z]
                s = s + p * mu_last[new_index_val_x, new_index_val_y, new_index_val_z]
            mu_last = s
            mu = mu + mu_last
        mu = mu / self.trajectory_length
        # mu = mu / n_time
        state_visitation = mu_last * state_space_model_features
        print "State Visitation Frequency calculated."
        return np.sum(state_visitation.reshape(2, 11 * 11 * 11), axis=1), policy
        # return mu_last, policy, state_space_model_features
        # state_visitation = mu_last * self.f
        # print "State Visitation Frequency calculated."
        # return np.sum(state_visitation.reshape(2, 11 * 11 * 11), axis=1), policy







