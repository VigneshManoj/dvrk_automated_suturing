import numpy as np
import matplotlib.pyplot as plt
from create_state_space import RobotMarkovModel
from max_ent_irl import MaxEntIRL

def main(discount, n_trajectories, epochs, learning_rate, weights):
    robot_mdp = RobotMarkovModel()
    irl = MaxEntIRL()
    model_rot_par_r, model_rot_par_p, model_rot_par_y, \
    model_end_pos_x, model_end_pos_y, model_end_pos_z = robot_mdp.state_space_model()

    # traj_features_array, traj_rewards_array = robot_mdp.trajectories_features_rewards_array(weights)
    traj_features_array = robot_mdp.trajectories_features_array()
    # print traj_features_array
    # print traj_rewards_array
    # print model_rot_par_r
    state_traj_array, action_traj_array = robot_mdp.trajectories_data()
    action_set = robot_mdp.create_action_set_func()
    reward_function = irl.max_ent_irl(traj_features_array, action_set, discount, state_traj_array,
                                      action_traj_array, epochs, learning_rate)


if __name__ == '__main__':
    rand_weights = np.random.rand(1, 2)
    main(0.01, 20, 200, 0.01, rand_weights)
