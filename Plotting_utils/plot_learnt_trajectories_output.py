import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Directory were the states visited by the user is stored
    file_dir_user_traj = "/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/formatplot.csv"
    # Directory where the states visited by the learnt policy is stored
    policy_dir = '/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/different_policies/'
    # Load the states visited by a specific policy (starting state  [-0.01, 0.055, -0.135] )
    # If you change the npy file, different policies output can be seen
    visit_states_policy = np.load(policy_dir + 'output3.npy')
    visit_states_policy = visit_states_policy.reshape(int(len(visit_states_policy)/3), 3)
    # For a specific policy, Directory where the files for different initial states is stored
    diff_state_dir = '/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11'
    # Load arrays of states visited after starting from different initial positions
    # 1 [-0.005, 0.055, -0.135]
    diff_init_state1 = np.load(diff_state_dir+'/random_points_output/output4.npy')
    diff_init_state1 = diff_init_state1.reshape(int(len(diff_init_state1)/3), 3)
    # 2 [-0.01, 0.055, -0.13]
    diff_init_state2 = np.load(diff_state_dir+'/random_points_output/output5.npy')
    diff_init_state2 = diff_init_state2.reshape(int(len(diff_init_state2)/3), 3)
    # 3 [-0.005, 0.055, -0.13]
    diff_init_state3 = np.load(diff_state_dir+'/random_points_output/output6.npy')
    diff_init_state3 = diff_init_state3.reshape(int(len(diff_init_state3)/3), 3)
    # 4 [-0.005, 0.005, -0.13]
    diff_init_state4 = np.load(diff_state_dir+'/random_points_output/output7.npy')
    diff_init_state4 = diff_init_state4.reshape(int(len(diff_init_state4)/3), 3)
    # 5 [-0.01, 0.06, -0.13]
    diff_init_state5 = np.load(diff_state_dir+'/random_points_output/output8.npy')
    diff_init_state5 = diff_init_state5.reshape(int(len(diff_init_state5)/3), 3)
    # 6 [-0.01, 0.06, -0.135]
    diff_init_state6 = np.load(diff_state_dir+'/random_points_output/output9.npy')
    diff_init_state6 = diff_init_state6.reshape(int(len(diff_init_state6)/3), 3)
    # 7 [-0.005, 0.06, -0.135]
    diff_init_state7 = np.load(diff_state_dir+'/random_points_output/output10.npy')
    diff_init_state7 = diff_init_state7.reshape(int(len(diff_init_state7)/3), 3)
    # 8 [-0.025, 0.055, -0.135]
    diff_init_state8 = np.load(diff_state_dir+'/random_points_output/output11.npy')
    diff_init_state8 = diff_init_state8.reshape(int(len(diff_init_state8)/3), 3)
    # 9 [-0.025, 0.05, -0.135]
    diff_init_state9 = np.load(diff_state_dir+'/random_points_output/output12.npy')
    diff_init_state9 = diff_init_state9.reshape(int(len(diff_init_state9)/3), 3)
    # 10 [-0.025, 0.06, -0.135]
    diff_init_state10 = np.load(diff_state_dir+'/random_points_output/output13.npy')
    diff_init_state10 = diff_init_state10.reshape(int(len(diff_init_state10)/3), 3)

    # States visited by the user
    user_visited_states = pd.read_csv(file_dir_user_traj).to_numpy()
    # Plot the graph
    plt.rcParams.update({'font.size': 12.5})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("Comparison between Learnt RL Trajectory and Operator Trajectory for Expanded Grid")
    # ax.set_title("Learnt RL Trajectory from different Initial States")
    ax.set_xlabel('X(m)', labelpad=10)
    ax.set_ylabel('Y(m)', labelpad=10)
    ax.set_zlabel('Z(m)', labelpad=10)
    plt.xticks([-0.03, -0.02, -0.01, 0, 0.010])
    plt.yticks([0.035, 0.04, 0.045, 0.05, 0.055])
    ax.set_zticks([-0.12, -0.124, -0.128, -0.132, -0.136])
    ax.tick_params(axis='X', labelsize=10)
    ax.tick_params(axis='Y', labelsize=10)
    ax.tick_params(axis='Z', labelsize=10)
    # Set the colors for RL and user trajectory
    color_rl = 'r'
    color_user = 'g'

    # Plot outputs from different learnt policies
    # ax.plot(visit_states_policy[:, 0], visit_states_policy[:, 1], visit_states_policy[:, 2],
    #         color=color_rl, linewidth=3)
    # ax.plot(user_visited_states[:, 0], user_visited_states[:, 1], user_visited_states[:, 2],
    #         linestyle='--', color=color_user, linewidth=3)

    # If required to plot the output for expanded grid (first one starting from same position as user
    # and second starting point is shifted by some value from user trajectory
    # ax.plot(visit_states_policy[:, 0], visit_states_policy[:, 1], visit_states_policy[:, 2],
    #         color=color_rl, linewidth=3)
    # ax.plot(user_visited_states[:, 0], user_visited_states[:, 1], user_visited_states[:, 2], linestyle='--',
    #         color=color_user, linewidth=3)
    # ax.plot(diff_init_state8[:, 0]+0.005, diff_init_state8[:, 1]+0.05, diff_init_state8[:, 2],
    #         color=color_rl, linewidth=3)
    # ax.plot(user_visited_states[:, 0]+0.005, user_visited_states[:, 1]+0.05, user_visited_states[:, 2],
    #         linestyle='--', color=color_user, linewidth=3)
    # ax.legend(["RL Learnt Trajectory", "Expert Trajectory"], loc=0, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

    # If want to plot all different initial states output in one graph:
    # ax.plot(diff_init_state1[:, 0], diff_init_state1[:, 1], diff_init_state1[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state2[:, 0], diff_init_state2[:, 1], diff_init_state2[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state3[:, 0], diff_init_state3[:, 1], diff_init_state3[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state4[:, 0], diff_init_state4[:, 1], diff_init_state4[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state5[:, 0], diff_init_state5[:, 1], diff_init_state5[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state6[:, 0], diff_init_state6[:, 1], diff_init_state6[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state7[:, 0], diff_init_state7[:, 1], diff_init_state7[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state8[:, 0], diff_init_state8[:, 1], diff_init_state8[:, 2], color=color_rl, linewidth=3)
    # ax.plot(diff_init_state9[:, 0], diff_init_state9[:, 1], diff_init_state9[:, 2], color=color_rl, linewidth=3)
    # ax.plot(user_visited_states[:, 0], user_visited_states[:, 1], user_visited_states[:, 2], linestyle='--',
    #         color=color_user, linewidth=3)
    # ax.legend(["Initial State 1", "Initial State 2", "Initial State 3", "Initial State 4",
    #           "Initial State 5", "Initial State 6", "Initial State 7", "Initial State 8", "Initial State 9",
    #           "User's Trajectory"], loc=0)

    # Save the plots
    plt.savefig(policy_dir + 'states_output3.png')
    # plt.savefig(diff_state_dir + '/random_points_output/different_states_output10.png')
    plt.show()