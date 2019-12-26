import numpy as np
import numpy.random as rn
import math
from robot_markov_model import RobotMarkovModel
from robot_state_utils import RobotStateUtils
from q_learning import q_learning, optimal_policy_func

class MaxEntIRL:

    def __init__(self, trajectory_length):
        # super(MaxEntIRL, self).__init__(11)
        self.trajectory_length = trajectory_length

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, grid_size, sum_trajectory_features, feature_array_all_trajectories, discount,
                    n_trajectories, epochs, learning_rate):
        # Finds the total number of states and dimensions of the list of features array
        total_states = len(feature_array_all_trajectories[0])
        d_states = len(feature_array_all_trajectories[0][0])
        # print "n states and d states ", n_states, d_states
        # Initialize with random weights based on the dimensionality of the states
        weights = np.random.rand(1, d_states)
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(sum_trajectory_features, n_trajectories)
        # Gradient descent on alpha
        for i in range(epochs):
            print "Epoch running is ", i
            # Multiplies the features with randomized alpha value, size of output Ex: dot(449*2, 2x1)
            optimal_policy, state_features, expected_svf = self.find_expected_svf(grid_size, weights, discount,
                                                                                  total_states, learning_rate)
            grad = feature_expectations.reshape(d_states, 1) - state_features.dot(expected_svf).reshape(d_states, 1)

            weights += learning_rate * np.transpose(grad)
            print "weights is ", weights
        # Compute the reward of the trajectory based on the weights value calculated
        trajectory_reward = np.dot(feature_array_all_trajectories[0][0:total_states], (weights.reshape(d_states, 1)))

        return trajectory_reward, weights

    def find_feature_expectations(self, trajectory_features, n_trajectories):
        # Takes the sum of all the expert features as input
        # Divide by the number of trajectories data available
        feature_expectations = trajectory_features/n_trajectories
        # Return the expert data feature expectations
        return feature_expectations

    def find_expected_svf(self, grid_size, weights, discount, total_states, learning_rate):
        # Creates object of robot markov model
        robot_mdp = RobotMarkovModel()
        # Returns the state and action values of the state space being created
        state_values_from_trajectory, _ = robot_mdp.return_trajectories_data()
        # Finds the terminal state value from the expert trajectory
        # 0 indicates that the first trajectory data is being used
        # total_states indicates that the last value of that trajectory data should be used as terminal state
        terminal_state_val_from_trajectory = state_values_from_trajectory[0][total_states-1]
        # Pass the gridsize required
        # To create the state space model, we need to run the below commands
        env_obj = RobotStateUtils(grid_size, weights, discount, terminal_state_val_from_trajectory)
        states = env_obj.create_state_space_model_func()
        action = env_obj.create_action_set_func()
        # print "State space created is ", states
        P_a = env_obj.get_transition_mat_deterministic()
        # print "P_a is ", P_a
        # Calculates the reward and feature for the trajectories being created in the state space
        rewards = []
        state_features = []
        for i in range(len(states)):
            r, f = env_obj.reward_func(states[i][0], states[i][1], states[i][2], weights)
            rewards.append(r)
            state_features.append(f)
        # print "rewards is ", rewards
        # value, policy = env_obj.value_iteration(rewards)
        policy = optimal_policy_func(states, action, env_obj, weights, learning_rate, discount)
        # policy = np.random.randint(27, size=1331)
        print "policy is ", policy
        # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
        expert_trajectory_states, _ = robot_mdp.return_trajectories_data()
        expected_svf = env_obj.compute_state_visitation_frequency(expert_trajectory_states, policy)
        # Formats the features array in a way that it can be multiplied with the svf values
        state_features = np.array([state_features]).transpose().reshape((len(state_features[0]), len(state_features)))
        print "svf is ", expected_svf
        print "svf shape is ", expected_svf.shape

        # Returns the policy, state features of the trajectories considered in the state space and expected svf
        return policy, state_features, expected_svf





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







