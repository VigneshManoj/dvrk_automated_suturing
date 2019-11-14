import numpy as np
import math
import features
# import matplotlib.pyplot as plt
import util_func
import sys

# n_iterations = int(sys.argv[1])
# rl_iter = int(sys.argv[2])
# svf_iter = int(sys.argv[3])
n_iterations = 1
rl_iter = 2
svf_iter = 2
print rl_iter, svf_iter
trajectories  = np.load('trajectory_class_1.npz')
state_trajectories = trajectories['state']
action_trajectories = trajectories['action']
n_traj = len(state_trajectories)

weights = np.random.rand(1,2)
print weights
Z = np.empty([0,1])
trajectories_probability = np.empty([len(state_trajectories), 1], dtype = 'float32')
for n in range(0,n_iterations):
    print "Iteration: ", n
    trajectories_reward = []
    trajectories_features = []
    for state_trajectory in state_trajectories:
        trajectory_reward = np.zeros([1,1], dtype = 'float32')
        trajectory_features = np.zeros([2,1], dtype = 'float32')
        for iter in range(0, state_trajectory.shape[0]):
            x = np.atleast_2d(state_trajectory[iter,0])
            y = np.atleast_2d(state_trajectory[iter,1])
            vtheta = np.atleast_2d(state_trajectory[iter,2])
            speed = np.atleast_2d(state_trajectory[iter,3])
            r, f = features.reward(x, y, vtheta, speed,weights)
            trajectory_reward = trajectory_reward +r
            trajectory_features = trajectory_features + np.vstack((f[0],f[1]))
        trajectories_reward.append(trajectory_reward)
        trajectories_features.append(trajectory_features)
    # print trajectory_features
    # print len(trajectories_reward)
    trajectories_probability = np.exp(trajectories_reward)
    feature_state, policy = cudatrial.get_policy(weights, rl_iter, svf_iter)
    # print sum(feature_state.reshape(301*301*101*11,1))
    Z = np.vstack((Z, sum(trajectories_reward)))
    # # trajectories_probability.reshape((len(trajectories_reward),1))
    # L=np.vstack((L,sum(trajectories_reward)/n_traj - np.log(Z)))
    # # if L[n]<L[n-1]:
    # #     break
    #
    grad_L = sum(trajectories_features)/n_traj- feature_state.reshape(2,1)
    print grad_L.shape
    #
    weights = weights + 0.005*np.transpose(grad_L)
    print Z[n]
np.save('final_policy', policy)
np.save('final_weights', weights)
print "Weights are:", weights
# print "Likelihood is :", L
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(Z)
# plt.show()
