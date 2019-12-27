import numpy as np
import matplotlib.pyplot as plt
from robot_markov_model import RobotMarkovModel
from max_ent_irl import MaxEntIRL

def main(grid_size, discount_factor, epochs, learning_rate):
    # Creates an object for using the RobotMarkovModel class
    robot_mdp = RobotMarkovModel()
    # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
    trajectories = robot_mdp.generate_trajectories()
    # Finds the length of the trajectories data
    n_trajectories = len(trajectories)
    # Initialize the IRL class object, provide trajectory length as input, currently its value is 1
    irl = MaxEntIRL(n_trajectories, grid_size)
    # Calculates the reward function based on the Max Entropy IRL algorithm
    reward, weights = irl.max_ent_irl(trajectories, discount_factor, n_trajectories, epochs, learning_rate)

    print "r is ", reward
    # print "r shape ", reward.shape
    print "weights is ", weights
    # print "policy is ", policy[0][0]

if __name__ == '__main__':
    # Epochs indicates the number of times gradient iteration needs to be run
    main(grid_size=3, discount_factor=0.9, epochs=150, learning_rate=0.01)
