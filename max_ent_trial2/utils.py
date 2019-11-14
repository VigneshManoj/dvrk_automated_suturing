import numpy as np
import numba as nb
import math
import time as t
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from numba import vectorize
import concurrent.futures
# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
import features
# import scipy.sparse as sp


class mdp(object):
    def __init__(self, init_joint_angles):
        self.joints = init_joint_angles
        self.action_set = []
        self.gamma = 0.9
        self.beta =  0.75

        # for j1 in [-0.01,0,0.01]:
        #     for j2 in [-0.01,0,0.01]:
        #         for j3 in [-0.1, 0, 0.1]:
        #             # for j4 in [-0.01, 0, 0.01]:
        #             self.action_set.append(np.array([j1,j2,j3,0]))


        self.projection_matrix =np.matrix([[1/math.tan(np.pi/12)*744/1301, 0,0,0],
                         [0, 1/math.tan(np.pi/12), 0,0],
                         [0, 0, -(100+0.1)/(100-0.1), -1],
                         [0, 0, -2*(100*0.1)/(100-0.1), 0]])

        self.T_base_to_rcm = np.matrix([[0,-math.sin(np.pi/3),math.cos(np.pi/3),3+5*math.cos(np.pi/3)],
                         [1,0,0,0],
                         [0,math.cos(np.pi/3),math.sin(np.pi/3),6+5*math.sin(np.pi/3)],
                         [0,0,0,1]])
        self.modelViewAdjusted = np.matrix([[0, 0, 1, 0],
                                           [1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])
        self.modelViewMatrix = self.getModelViewMatrix(self.joints)

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
    def getModelViewMatrix(self, joints):
        modelViewMatrix = np.transpose(np.linalg.inv(self.T_base_to_rcm*self.forwardKinematics(joints)*self.modelViewAdjusted))
        return modelViewMatrix
    def update_joints(action):
        self.joints = self.joints + action

    # def get_next_state(self,x, y,action):
    #     joints = self.joints + action
    #     modelViewMatrix = self.getModelViewMatrix(joints)
    #     screen_pos = np.transpose(modelViewMatrix*self.projection_matrix)*[[0],[0],[0],[1]]
    #     state_position = [screen_pos[0,0]/(screen_pos[3,0]),screen_pos[1,0]/(screen_pos[3,0])]
    #     next_state = [state_position[0], state_position[1], 0,0,0]
    #     print state_position
    #     return next_state

    # @nb.jit
    def get_next_state(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z, action):
        
        left_end_effector = self.left_end_effector + action
        # print delta
        delta_s = (delta-cur_state*delta[3,:])
        next_state  = cur_state + delta_s
        return next_state[0,0], next_state[1,0], vtheta, s
    '''

    def get_next_state(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z, action):
        curr_state = np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z])
        next_state_val = curr_state + action
        return next_state_val

    # def reward(self,state):
    #     x = state[0]
    #     y = state[1]
    #     # r = x
    #     r = np.exp(-(x**2+y**2)/(0.1**2))
    #     return r
    #
    # def value(self, state, iter):
    #     if iter == 0:
    #         val = self.reward(state)
    #     else:
    #         val = 0
    #         for action in self.action_set:
    #             val =   val + self.action_value(state,action,iter)
    #     return val
    #
    # def action_value(self, state, action, iter):
    #     if iter == 0:
    #         val = self.reward(state)
    #     else:
    #         next_state = self.get_next_state(state,action)
    #         val = self.reward(state) + self.gamma*self.value(next_state, iter-1)
    #     return val
    #
    # def calculate_z(self,state, iter):
    #     Z=[]
    #     for action in self.action_set:
    #         Z.append(np.exp(self.beta*self.action_value(state, action, iter)))
    #     # print sum(Z)
    #     return Z
    #
    # def policy(self,state, iter):
    #     Z = self.calculate_z(state,iter)
    #     pol = Z/sum(Z)
    #     # print Z
    #     # for action in self.action_set:
    #     #     pol.append([np.exp(self.beta*self.action_value(state, action, iter))]/Z)
    #     return pol

# @vectorize(nb.types.UniTuple(nb.int64[:],3)(nb.float32[:],3), target='parallel')


# Created indices function. It basically reads the value and based on the value it rounds it of to the nearest state
# space value. For RPY values it rounds it off to the nearest 0.01 value and for XYZ pos values it rounds it off to
# the nearest 0.001 value
def get_indices(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):

    end_pos_x = np.round(end_pos_x, 2)
    end_pos_y = np.round(end_pos_y, 2)
    end_pos_z = np.round(end_pos_z, 2)

    end_pos_x[end_pos_x <= -1.5] = -1.5
    end_pos_y[end_pos_y <= -1.5] = -1.5
    end_pos_z[end_pos_z <= -1.5] = -1.5

    end_pos_x[end_pos_x >= 1.5] = 1.5
    end_pos_y[end_pos_y >= 1.5] = 1.5
    end_pos_z[end_pos_z >= 1.5] = 1.5

    index_end_pos_x = (end_pos_x + 1.5)*100
    index_end_pos_y = (end_pos_y + 1.5)*100
    index_end_pos_z = (end_pos_z + 1.5)*100

    index_rot_par_r = (rot_par_r+math.pi)*50/math.pi + 0.5
    index_rot_par_p = (rot_par_p+math.pi)*50/math.pi + 0.5
    index_rot_par_y = (rot_par_y+math.pi)*50/math.pi + 0.5

    index_end_pos_x = index_end_pos_x.astype(int)
    index_end_pos_y = index_end_pos_y.astype(int)
    index_end_pos_z = index_end_pos_z.astype(int)

    index_rot_par_r = index_rot_par_r.astype(int)
    index_rot_par_p = index_rot_par_p.astype(int)
    index_rot_par_y = index_rot_par_y.astype(int)

    return index_rot_par_r, index_rot_par_p, index_rot_par_y, index_end_pos_x, index_end_pos_y, index_end_pos_z


# The reward function which is basically adding both the feature1 and feature2 and then using that to calculate the
# reward
@vectorize(['float64(float64, float64, float64, float64)'], target='parallel')
def reward(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):

    r = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2) \
          + np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
    return r


def main_loop(action):
    new_state_rot_par_r, new_state_rot_par_p, new_state_rot_par_y, new_state_end_pos_x, new_state_end_pos_y, \
    new_state_end_pos_z = model.get_next_state(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z, action)

    new_index_rot_par_r, new_index_rot_par_p, new_index_rot_par_y, new_index_end_pos_x, new_index_end_pos_y, \
    new_index_end_pos_z = get_indices(new_state_rot_par_r, new_state_rot_par_p, new_state_rot_par_y,
                                      new_state_end_pos_x, new_state_end_pos_y, new_state_end_pos_z)

    q = r[index_rot_par_r, index_rot_par_p, index_rot_par_y, index_end_pos_x, index_end_pos_y, index_end_pos_z] + \
        0.9*v[new_index_rot_par_r, new_index_rot_par_p, new_index_rot_par_y, new_index_end_pos_x, new_index_end_pos_y,
              new_index_end_pos_z]
    p = np.exp(0.75*q)
    return q, p

def initial_loop(action):
    # print "Calculating q..."
    q = r[index_rot_par_r, index_rot_par_p, index_rot_par_y, index_end_pos_x, index_end_pos_y, index_end_pos_z]
    # print "Calculating p..."
    p = np.exp(0.75*q)
    return q, p

def get_policy(weights, n_iter, n_time):
    global model
    global r, v, index_x, index_y, index_vel_theta, index_speed, state_x, state_y, state_vel_theta, state_speed
    global rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z
    global index_rot_par_r, index_rot_par_p, index_rot_par_y, index_end_pos_x, index_end_pos_y, index_end_pos_z
    global state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z
    model = mdp(np.array([1, 0, 0, 0, 0, 0], dtype='float64'))

    end_pos_x = np.linspace(-1.5, 1.5, 301, dtype='float64')
    end_pos_y = np.linspace(-1.5, 1.5, 301, dtype='float64')
    end_pos_z = np.linspace(-1.5, 1.5, 301, dtype='float64')

    rot_par_r = np.linspace(-math.pi, math.pi, 101, dtype='float64')
    rot_par_p = np.linspace(-math.pi, math.pi, 101, dtype='float64')
    rot_par_y = np.linspace(-math.pi, math.pi, 101, dtype='float64')

    print 'Creating state space...'
    state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z = \
        np.meshgrid(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z)
    print 'State space created.'

    # plot_x = np.linspace(-1.5,1.5,21, dtype = 'float32')
    # plot_z = np.linspace(0.9, 1, 3, dtype = 'float32')
    # plot_xv, plot_yv, plot_zv = np.meshgrid(plot_x, plot_x, plot_z)
    # print xv.shape

    action_set= []
    for rot_par_r in [-0.01, 0, 0.01]:
        for rot_par_p in [-0.01, 0, 0.01]:
            for rot_par_y in [-0.01, 0, 0.01]:
                for end_pos_x in [-0.001, 0, 0.001]:
                    for end_pos_y in [-0.001, 0, 0.001]:
                        for end_pos_z in [-0.001, 0, 0.001]:
                            action_set.append(
                                np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z]))

    # print len(action_set)
    # r = reward(state_x, state_y, state_vel_theta, state_speed)

    r, f = features.reward(state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y,
                           state_end_pos_z, weights)


    index_rot_par_r, index_rot_par_p, index_rot_par_y, index_end_pos_x, index_end_pos_y, index_end_pos_z = \
        get_indices(state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z)

    policy = []
    for iter in range(0, n_iter):
        action_value = []
        policy =[]
        print "Policy Iteration:", iter
        # start_time = t.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            if iter == 0:
                func = initial_loop
            else:
                func = main_loop
            for q, p in executor.map(func, action_set):
                action_value.append(q)
                policy.append(p)

        print "Evaluating Policy..."
        policy = policy/sum(policy)
        v = sum(policy*action_value)
        # end_time = t.time()
        # print end_time-start_time

    # mu = np.empty([301,301,101,11])
    print "Final Policy evaluated."
    print "Calulating State Visitation Frequency..."
    mu = np.exp(-(state_x+0.15)**2/0.25**2)*np.exp(-(state_y-0.27)**2/0.5**2)*np.exp(0.004*state_speed)
    mu_reshape = np.reshape(mu, [301*301*101*11,1])
    mu = mu/sum(mu_reshape)
    mu_last = mu
    print "Initial State Frequency calculated..."
    for time in range(0,n_time):
        s = np.zeros([301,301,101,11])
        for act_index, action in enumerate(action_set):

            new_state_x, new_state_v, new_state_vel_theta, new_state_speed = model.get_next_state(state_x, state_y, state_vel_theta, state_speed, action)

            new_index_x, new_index_y, new_index_vel_theta, new_index_speed = get_indices(new_state_x, new_state_v, new_state_vel_theta, new_state_speed)

            p = policy[act_index, index_x, index_y, index_vel_theta, index_speed]
            s = s + p*mu_last[new_index_x, new_index_y, new_index_vel_theta, new_index_speed]
        mu_last = s
        mu = mu+mu_last
    mu = mu/n_time
    state_visitation = mu_last*f
    print "State Visitation Frequency calculated."
    return np.sum(state_visitation.reshape(2, 301*301*101*11), axis=1), policy
    # return f, policy, mu
    # mu_reshape = np.reshape(mu_normalized, [301*301*101*11,1])
    # print sum(mu_reshape)
