import numpy as np
import matplotlib.pyplot as plt
from robot_markov_model import RobotMarkovModel
from max_ent_irl import MaxEntIRL

def main(discount, epochs, learning_rate, weights, trajectory_length):
    robot_mdp = RobotMarkovModel()
    irl = MaxEntIRL(trajectory_length)

    # trajectory_features_array, trajectory_rewards_array = robot_mdp.trajectories_features_rewards_array(weights)
    trajectory_features_array, complete_features_array = robot_mdp.trajectories_features_array()
    # print "traj array ", trajectory_features_array.shape, trajectory_features_array
    # print "complete array ", complete_features_array.shape, complete_features_array
    # print trajectory_rewards_array
    # print model_rot_par_r
    state_trajectory_array, action_trajectory_array = robot_mdp.trajectories_data()
    n_trajectories = len(state_trajectory_array)
    action_set = robot_mdp.create_action_set_func()
    reward = irl.max_ent_irl(trajectory_features_array, complete_features_array, action_set, discount,
                             n_trajectories, epochs, learning_rate)

    print "r is ", reward

if __name__ == '__main__':
    rand_weights = np.random.rand(1, 2)
    trajectory_length = 1
    main(0.01, 200, 0.01, rand_weights, trajectory_length)
