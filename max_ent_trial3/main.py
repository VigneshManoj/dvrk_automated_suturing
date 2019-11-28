import numpy as np
import matplotlib.pyplot as plt
from create_state_space import RobotMarkovModel


def main(discount, n_trajectories, epochs, learning_rate, weights):
    robot_space = RobotMarkovModel()
    # robot_mdp = RobotMarkovModel()
    model_rot_par_r, model_rot_par_p, model_rot_par_y, \
    model_end_pos_x, model_end_pos_y, model_end_pos_z = robot_space.state_space_model()

    # state_trajectories, action_trajectories = robot_space.trajectories_data()
    traj_features_array, traj_rewards_array = robot_space.trajectories_features_array(weights)
    print traj_features_array
    print traj_rewards_array
    print model_rot_par_r
    # reward_function = maxent.irl(feature_matrix, gw.n_actions, discount,
    #               gw.transition_probability, trajectories, epochs, learning_rate)


if __name__ == '__main__':
    rand_weights = np.random.rand(1, 2)
    main(0.01, 20, 200, 0.01, rand_weights)
