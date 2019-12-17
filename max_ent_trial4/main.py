import numpy as np
import matplotlib.pyplot as plt
from robot_markov_model import RobotMarkovModel
from max_ent_irl import MaxEntIRL

def main(discount, epochs, learning_rate, n_policy_iter):
    # Creates an object for using the RobotMarkovModel class
    robot_mdp = RobotMarkovModel()
    # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
    sum_trajectory_features, feature_array_all_trajectories = robot_mdp.generate_trajectories()
    # Finds the length of the trajectories data
    n_trajectories = len(feature_array_all_trajectories)
    #  print "number of trajectories is ", n_trajectories
    # Initialize the IRL class object, provide trajectory length as input, currently its value is 1
    irl = MaxEntIRL(n_trajectories)
    # Calculates the reward function based on the Max Entropy IRL algorithm
    reward, weights = irl.max_ent_irl(sum_trajectory_features, feature_array_all_trajectories, discount,
                                    n_trajectories, epochs, learning_rate, n_policy_iter)

    # print "r is ", reward
    # print "r shape ", reward.shape
    # print "alpha is ", weights
    # print "policy is ", policy[0][0]

if __name__ == '__main__':
    # The number of times policy iteration needs to be run
    n_policy_iter = 3
    main(discount=0.9, epochs=200, learning_rate=0.01, n_policy_iter=n_policy_iter)
