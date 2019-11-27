import numpy as np
import matplotlib.pyplot as plt
from create_state_space import RobotStateSpace
from robot_mdp import RobotMarkovModel


def main(discount, n_trajectories, epochs, learning_rate, file_path):
    robot_space = RobotStateSpace()
    robot_mdp = RobotMarkovModel()
    model_rot_par_r, model_rot_par_p, model_rot_par_y, \
    model_end_pos_x, model_end_pos_y, model_end_pos_z = robot_space.state_space_model()

    state_trajectories, action_trajectories = robot_space.trajectories_data(file_path)
    features_array = robot_mdp.features_array(model_rot_par_r, model_rot_par_p, model_rot_par_y,
                                              model_end_pos_x, model_end_pos_y, model_end_pos_z)
    # print "feature matrix is ", feature_matrix
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    r = maxent.irl(feature_matrix, gw.n_actions, discount,
                   gw.transition_probability, trajectories, epochs, learning_rate)


if __name__ == '__main__':
    # Provide the location of the trajectory data to be used
    trajectory_file_path = "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data_without_norm.csv"
    main(0.01, 20, 200, 0.01, trajectory_file_path)
