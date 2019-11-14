import numpy as np
import math
from numba import vectorize

class mdp(object):
    def __init__(self, left_end_effector ):
        self.left_end_effector = left_end_effector
        self.action_set = []
        self.gamma = 0.9
        self.beta =  0.75

        # Need only six parameters to define a robot, velocity can be found out by subtracting two consecutive position values and dividing it by the time. So control only position of the robot.
        for rot_par_r in [-0.01,0,0.01]:
            for rot_par_p in [-0.01,0,0.01]:
                for rot_par_y in [-0.01,0,0.01]:
                    for end_pos_x in [-0.001, 0, 0.001]:
                        for end_pos_y in [-0.001, 0, 0.001]:
                            for end_pos_z in [-0.001, 0, 0.001]:
                                    self.action_set.append(np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z]))
    '''
    def Compute_DH_Matrix(self,alpha, a , theta, d):
        DH_matrix=np.matrix([[math.cos(theta), -math.sin(theta), 0, a],
                            [math.sin(theta)*math.cos(alpha), math.cos(theta)*math.cos(alpha), -math.sin(alpha), -d*math.sin(alpha)],
                            [math.sin(theta)*math.sin(alpha), math.cos(theta)*math.sin(alpha), math.cos(alpha), d*math.cos(alpha)],
                            [0, 0, 0, 1]])
        return DH_matrix

    def forwardKinematics(self, joints):
        DH=np.matrix([[np.pi/2, 0.0000, joints[0]+np.pi/2, 0.0000],
                     [-np.pi/2, 0.0000, joints[1]-np.pi/2, 0.0000],
                     [np.pi/2, 0.0000, 0.0, joints[2]],
                     [0.0000, 0.0000, joints[3], 0]])
        FK=np.identity(4)
        i=0
        self.Compute_DH_Matrix(DH[0,0],DH[0,1],DH[0,2],DH[0,3])
        for i in range(0, 4):
            FK=FK * self.Compute_DH_Matrix(DH[i,0],DH[i,1],DH[i,2],DH[i,3])
        FK = FK * np.matrix([[0,0,-1,0],[0,-1,0,0],[-1,0,0,0],[0,0,0,1]])
        return FK

    def update_joints(self, action):
        self.joints = self.joints + action
    '''

    # Adds the action value and provides the next state of the robot
    def get_next_state(self, state, action):
        left_end_effector = self.left_end_effector + action
        return left_end_effector


    # Wrote the reward function to be used as exponential of sum of individually squared value divided by
    # # the variance value
    def reward(self,state):
        rot_par_r = state[0]
        rot_par_p = state[1]
        rot_par_y = state[2]
        end_pos_x = state[3]
        end_pos_y = state[4]
        end_pos_z = state[5]

        r = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2)
        return r

    def value(self, state, iter):
        if iter == 0:
            val = self.reward(state)
        else:
            val = 0
            for action in self.action_set:
                val = val + self.action_value(state,action,iter)
        return val

    def action_value(self, state, action, iter):
        if iter == 0:
            val = self.reward(state)
        else:
            next_state = self.get_next_state(state,action)
            val = self.reward(state) + self.gamma*self.value(next_state, iter-1)
        return val

    def calculate_z(self,state, iter):
        Z=[]
        for action in self.action_set:
            Z.append(np.exp(self.beta*self.action_value(state, action, iter)))
        # print sum(Z)
        return Z

    def policy(self,state, iter):
        Z = self.calculate_z(state, iter)
        pol = Z/sum(Z)
        # print Z
        # for action in self.action_set:
        #     pol.append([np.exp(self.beta*self.action_value(state, action, iter))]/Z)
        return pol


    # def irl(mdp, trajectories, features, n_iter, step_size, k):
#
#     '''
#     mdp : defines the Markov Decision Process of what is state space, action space, transition probabilities and discount factor
#     trajectories: defines the set of trajectory([(s1,a1),..(st,at)]  sequences, where t is the length of trajectory)
#     features : defines set of feature functions for reward mapping
#     n_iter : number of iterations to tune the feature weights
#     step_size : learning rate for feature weights
#     k : softmax activation constant
#     '''
#     feature_weights = np.random(features.number);
#
#     for t in range(0,n_iter):
#         reward = feature_weigths' * features.getVector(state)

if __name__ == '__main__':
    state = np.array([1,0,0,0,0], dtype=np.float32)
    # action = np.array([0.1,0,0,0])
    ecm = mdp(np.array([0,0,5,0], dtype=np.float32))
    Policy = ecm.policy(state, 2)
    action_index = Policy.argmax()
    action = ecm.action_set[action_index]
    print action
    # print max(Policy)
    # next_state = ecm.get_next_state(state,action)

    # print value(state, 0)
    # r = ecm.reward(state)
    # print r
