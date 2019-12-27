import numpy as np
import numpy.random as rn
import math
from robot_markov_model import RobotMarkovModel
from robot_state_utils import RobotStateUtils, q_learning, optimal_policy_func


class MaxEntIRL:

    def __init__(self, trajectory_length, grid_size):
        # super(MaxEntIRL, self).__init__(11)
        self.trajectory_length = trajectory_length
        self.grid_size = grid_size

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, trajectories, discount, n_trajectories, epochs, learning_rate):
        # Finds the total number of states and dimensions of the list of features array
        total_states = len(trajectories[0])
        d_states = len(trajectories[0][0])
        # print "n states and d states ", n_states, d_states
        feat_map = np.eye(self.grid_size ** 3)
        # Initialize with random weights based on the dimensionality of the states
        weights = np.random.uniform(size=(feat_map.shape[1],))
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(feat_map, trajectories)
        # print "feature array is ", feature_array_all_trajectories[0][0:total_states]
        # Gradient descent on alpha
        for i in range(epochs):
            print "Epoch running is ", i
            # Multiplies the features with randomized alpha value, size of output Ex: dot(449*2, 2x1)
            reward = np.dot(feat_map, weights)
            optimal_policy, expected_svf = self.find_expected_svf(reward, discount, trajectories, learning_rate)
            grad = feature_expectations - feat_map.T.dot(expected_svf)
            if i < 50:
                learning_rate = 0.1
            else:
                learning_rate = 0.01
            weights += learning_rate * np.transpose(grad)
            print "weights is ", weights
        # Compute the reward of the trajectory based on the weights value calculated
        trajectory_reward = np.dot(feat_map, weights)

        return trajectory_reward, weights


    def find_feature_expectations(self, feat_map, trajectories):
        feat_exp = np.zeros([feat_map.shape[1]])
        for episode in trajectories:
            # print "episode is ", episode
            for state_value in episode:
                # print "state value is ", state_value
                state_value_index = self.get_state_val_index(state_value)
                # print "feat map is ", feat_map[state_value_index, :]
                feat_exp += feat_map[state_value_index, :]
        feat_exp = feat_exp / len(trajectories)

        return feat_exp

    def find_expected_svf(self, reward, discount, trajectories, learning_rate):
        # Creates object of robot markov model
        # 0 indicates that the first trajectory data is being used
        total_states = len(trajectories[0])
        # total_states indicates that the last value of that trajectory data should be used as terminal state
        terminal_state_val_from_trajectory = trajectories[0][total_states-1]
        print "Terminal state value is ", terminal_state_val_from_trajectory
        # Pass the gridsize required
        # To create the state space model, we need to run the below commands
        env_obj = RobotStateUtils(self.grid_size, discount, terminal_state_val_from_trajectory)
        states = env_obj.create_state_space_model_func()
        action = env_obj.create_action_set_func()
        # print "State space created is ", states
        P_a = env_obj.get_transition_mat_deterministic()
        # print "P_a is ", P_a

        policy = optimal_policy_func(states, action, env_obj, reward, learning_rate, discount)
        # policy = np.random.randint(27, size=1331)
        print "policy is ", policy

        # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
        expected_svf = env_obj.compute_state_visitation_frequency(trajectories, policy)
        # Formats the features array in a way that it can be multiplied with the svf values
        print "svf is ", expected_svf
        # print "svf shape is ", expected_svf.shape

        # Returns the policy, state features of the trajectories considered in the state space and expected svf
        return policy, expected_svf

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.5) * pow(self.grid_size, 2)) + abs(
            (state_val[1] + 0.5) * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.5))
        return int(round(index_val * (self.grid_size - 1)))




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







