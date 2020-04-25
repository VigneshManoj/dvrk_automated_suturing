import numpy as np

from scipy.spatial.distance import euclidean, seuclidean, correlation, cityblock

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from fastdtw import fastdtw


def find_similarity(user, learnt, distance_type=cityblock):
    """
    distance_type = scipy.spatial.distance functions
    eg. euclidean, seuclidean, correlation, cityblock
    try out different ones to know the difference.
    cityblock is manhattan and requires each axis to be shown differently for a
    proper output.
    """

    distance, path = fastdtw(user, learnt, dist=distance_type)

    similarity = []

    for user_idx, learnt_idx in path:
        # IF CITYBLOCK
        similarity.append([np.round(distance_type(user[user_idx, 0], learnt[learnt_idx, 0]), 5),
                           np.round(distance_type(user[user_idx, 1], learnt[learnt_idx, 1]), 5),
                           np.round(distance_type(user[user_idx, 2], learnt[learnt_idx, 2]), 5)
                           ])
    # # FOR ANY OTHER
    # # UNCOMMENT

    # similarity.append(
    #   np.round(distance_type(user[user_idx],
    #                          learnt[learnt_idx,
    #                                 ]),
    #            5)
    # )

    print(similarity)

    # find the mean and variance over all x, y, z axes now

    return distance, path


def plot(user, learnt, path):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X(m)', labelpad=10)
    ax.set_ylabel('Y(m)', labelpad=10)
    ax.set_zlabel('Z(m)', labelpad=10)

    plt.xticks([-0.03, -0.02, -0.01, 0, 0.010])
    plt.yticks([0.035, 0.04, 0.045, 0.05, 0.055])
    ax.set_zticks([-0.12, -0.124, -0.128, -0.132, -0.136])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    color_rl = 'r'
    color_user = 'g'

    ax.plot(user[:, 0], user[:, 1], user[:, 2], color=color_user, linewidth=2)

    ax.plot(learnt[:, 0], learnt[:, 1], learnt[:, 2], color=color_rl, linewidth=2)

    for user_idx, learnt_idx in path:
        ax.plot([user[user_idx, 0], learnt[learnt_idx, 0]],
                [user[user_idx, 1], learnt[learnt_idx, 1]],
                [user[user_idx, 2], learnt[learnt_idx, 2]],
                color='k', linestyle='--', linewidth=1)

    plt.show()


if __name__ == '__main__':
    folder_dir = '/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/'
    user = np.load(folder_dir + 'user_traj.npy')
    learnt = np.load(folder_dir + 'learnt_traj.npy')

    distance, path = find_similarity(user, learnt, cityblock)
    plot(user, learnt, path)