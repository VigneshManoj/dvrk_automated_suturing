import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from robot_state_utils import RobotStateUtils
from robot_markov_model import RobotMarkovModel
from matplotlib.lines import Line2D


# Create the grid points for matplotlib
trajectories_z0 = [[0, 0, 0, 0] for i in range(1331)]
default_points = np.array([-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.010])

i = 0
# To create 3D points which can have different size
for z in default_points:
    for y in default_points:
        for x in default_points:
            trajectories_z0[i] = [x, y, z, 0.]
            i += 1

# Define the goal position (3D coordinate) of agent
goal = np.array([0.005, 0.055, -0.125])
# Create the Object of class which creates the state space and action space
env_obj = RobotStateUtils(11, 0.1, goal)
states = env_obj.create_state_space_model_func()
action = env_obj.create_action_set_func()
policy = np.zeros(len(states))
rewards = np.zeros(len(states))
index = env_obj.get_state_val_index(goal)
mdp_obj = RobotMarkovModel()
trajectories = mdp_obj.generate_trajectories()
visited_states_index_vals = np.zeros(len(trajectories[0]))
# Find the index values of the visited states
for i in range(len(trajectories[0])):
    visited_states_index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
for _, ele in enumerate(visited_states_index_vals):
    if ele == index:
        rewards[int(ele)] = 10
    else:
        rewards[int(ele)] = 1

# Weights here correspond to the size in matplotlib graph plot
# Weights can be based on the weights received from IRL output (npy file) or
# Weight can be based of the sparse reward created above
# If using IRL output, uncomment following 3 lines
# folder_dir = "/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/" \
#             "different_trial1/grid11_parallel/weights_grid11.npy"
# weights = np.load(folder_dir)
# For sparse rewards
weights = rewards
# Reshape the weights to a gridsize X gridsize X gridsize (in this case its 11)
Mat = weights.reshape((11, 11, 11), order='F')
mFlat = Mat.flatten()
# print("mflat is ", mFlat)
norm = mFlat/np.linalg.norm(mFlat)
# print("norm is ", norm)
# If want to threshold the weights output (ideal for sparse but not IRL output)
# norm_thresh = np.zeros(1331)
# for i in range(len(norm)):
#   if norm[i] >=0.2:
#     norm_thresh[i] = 1
#   else:
#     norm_thresh[i] = 0.1
#
# Print average of weight values
print("norm thresh is ", np.average(norm))
# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1331):
    if abs(norm[i]) >= 0.1:
        color = 'red'
    else:
        color = 'blue'
    trajectories_z0[i][3] = norm[i]
    ax.scatter(trajectories_z0[i][0], trajectories_z0[i][1], trajectories_z0[i][2], s=trajectories_z0[i][3] * 100,
               marker="o", color=color)

plt.rcParams.update({'font.size': 12.5})
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='z', labelsize=10)
# Set the values to dispaly in x,y,z axis
plt.xticks([-0.03, -0.02, -0.01, 0, 0.010])
plt.yticks([-0.03, -0.02, -0.01, 0, 0.010])
ax.set_zticks([-0.03, -0.02, -0.01, 0, 0.010])
# plt.zticks([-0.03, -0.02, -0.01, 0, 0.010])
# Create legend
legend_elements = [Line2D([0], [0], marker='o', color='r', label='Normalized Reward > 0.1', markersize=5),
                   Line2D([0], [0], marker='o', color='b', label='Normalized Reward < 0.1', markersize=5)]
ax.legend(handles=legend_elements, loc=1, bbox_to_anchor=(0.6, 0.6, 0.5, 0.5))
# Set X, Y, Z label values
ax.set_xlabel('X Position', labelpad=5)
ax.set_ylabel('Y Position', labelpad=5)
ax.set_zlabel('Z Position', labelpad=5)
# plt.title('Operator 1: Distribution of Reward Function for a custom trajectory')
plt.show()
save_fig_dir = "/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/" \
               "different_trial1/grid11_parallel/grid11_3d_plot_sparse"
fig.savefig(save_fig_dir)
