"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from read_write_joint_to_file import DataCollection
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from environment import Environment

if __name__ == '__main__':
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    # wind = 0.3
    # trajectory_length = 3*grid_size
    obj_data_collec = DataCollection(1, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_data_1_1000hz.csv")
    trajectories = obj_data_collec. data_parse_numpy()
    obj_environment = Environment(0.9)
    feature_matrix = obj_environment.feature_matrix()
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    r = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()