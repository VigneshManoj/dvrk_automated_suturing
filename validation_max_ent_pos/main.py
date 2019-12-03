import numpy as np
import matplotlib.pyplot as plt
from robot_markov_model import RobotMarkovModel
from max_ent_irl import MaxEntIRL

def main(discount, epochs, learning_rate, weights, trajectory_length, n_policy_iter):
    # Creates an object for using the RobotMarkovModel class
    robot_mdp = RobotMarkovModel()
    # Initialize the IRL class object, provide trajectory length as input, currently its value is 1
    irl = MaxEntIRL(trajectory_length)

    # trajectory_features_array, trajectory_rewards_array = robot_mdp.trajectories_features_rewards_array(weights)
    # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
    trajectory_features_array, complete_features_array = robot_mdp.trajectories_features_array()
    # print "traj array ", trajectory_features_array.shape, trajectory_features_array
    # print "complete array ", complete_features_array.shape, complete_features_array
    # print trajectory_rewards_array
    # print model_rot_par_r
    # Returns the state and actions spaces of the expert trajectory
    state_trajectory_array, action_trajectory_array = robot_mdp.trajectories_data()
    # Finds the length of the trajectories data
    n_trajectories = len(state_trajectory_array)
    # Calculates the reward function based on the Max Entropy IRL algorithm
    reward, alpha, policy = irl.max_ent_irl(trajectory_features_array, complete_features_array, discount,
                             n_trajectories, epochs, learning_rate, n_policy_iter, weights)

    print "r is ", reward
    print "alpha is ", alpha
    print "policy is ", policy[0][0]

if __name__ == '__main__':
    rand_weights = np.random.rand(1, 3)
    # The different kind of trajectories present in the user study
    trajectory_length = 1
    # The number of times policy iteration needs to be run
    n_policy_iter = 3
    main(0.9, 200, 0.01, rand_weights, trajectory_length, n_policy_iter)
