import numpy as np
import matplotlib.pyplot as plt
from create_state_space import RobotStateSpace

def main(discount, n_trajectories, epochs, learning_rate):

    trajectory_length = 3*grid_size
    state_space = RobotStateSpace()
    model_rot_par_r, model_rot_par_p, model_rot_par_y, \
    model_end_pos_x, model_end_pos_y, model_end_pos_z = state_space.state_space_model()

    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy)
    # print "trajectories is ", trajectories
    feature_matrix = gw.feature_matrix()
    # print "feature matrix is ", feature_matrix
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    r = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)


if __name__ == '__main__':
    main(0.01, 20, 200, 0.01)