import numpy as np
import numpy.random as rn
import math


class MaxEntIRL:
    def __init__(self, trajectory_length):
        self.reward = 0
        self.trajectory_length = trajectory_length

    def max_ent_irl(self, trajectory_features_array, complete_features_array, action_set, discount,
                             n_trajectories, epochs, learning_rate):
        n_states, d_states = complete_features_array.shape
        n_actions = len(action_set)
        # print "c feature", complete_features_array.shape
        # print "length of action set", len(action_set)
        alpha = rn.uniform(size=(d_states,))
        print "alpha is ", alpha
        feature_expectations = self.find_feature_expectations(trajectory_features_array)
        # Gradient descent on alpha.
        for i in range(epochs):
            # print("i: {}".format(i))
            self.reward = complete_features_array.dot(alpha)
            expected_svf = self.find_expected_svf(n_states, n_actions, discount, n_trajectories)
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

    def find_expected_svf(self, n_states, n_actions, discount, n_trajectories):
        # Trajectory length is calculated as follows:
        # Trajectory is basically list of all Traj1 Traj2 Traj3 Traj4 of user 1 and
        # similarly  Traj1 Traj2 Traj3 Traj4 for user 2
        # So basically inside trajectory is the list of all states visited
        # Outisde trajectory value would be the different trajectories collected from user (say trial1 trial2 etc)
        # So in this case currently, it is 1 in our case

        # policy = find_policy(n_states, r, n_actions, discount, transition_probability)
        policy = value_iteration.find_policy(n_states, n_actions, self.reward, discount)

        start_state_count = np.zeros(n_states)
        for trajectory in trajectories:
            start_state_count[trajectory[0, 0]] += 1
        p_start_state = start_state_count / n_trajectories

        expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
        for t in range(1, trajectory_length):
            expected_svf[:, t] = 0
            for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
                expected_svf[k, t] += (expected_svf[i, t - 1] *
                                       policy[i, j] *  # Stochastic policy
                                       transition_probability[i, j, k])

        return expected_svf.sum(axis=1)

    def calculate_optimal_policy_func(self):
        model_rot_par_r, model_rot_par_p, model_rot_par_y, \
        model_end_pos_x, model_end_pos_y, model_end_pos_z = robot_mdp.state_space_model()




