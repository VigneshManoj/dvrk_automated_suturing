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


class RobotMarkovModel(object):
    def __init__(self):
        # self.joints = init_joint_angles
        self.action_set = []
        self.gamma = 0.9
        self.beta = 0.75

        # Model here means the 3D cube being created
        # linspace limit values: limit_values_angle = [[-0.5, 0.5], [-0.234, -0.155], [0.28, 0.443]]
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 6 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.model_pos_x_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_pos_y_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_pos_z_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_rot_r_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        self.model_rot_p_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        self.model_rot_y_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
        # Change this value below based on number of points you select above, it basically assigns the indices values
        self.unit_linspace = np.linspace(0, 10, 11)
        # Initialize feature, reward and value function value to 0
        self.f = np.empty([11, 11, 11, 11, 11, 11])
        self.r = np.empty([11, 11, 11, 11, 11, 11])
        self.v = []
        self.integer_values = [0, 1, 2, 3, 4, 5]
        # The state values are being provided by the max_ent program
        self.state_values = np.zeros(6)
        # The created model state values
        self.model_state_values = np.zeros(6)
        self.model_index_values = np.zeros(6)
        self.action_set = []


# Created indices function. It basically reads the value and based on the value it rounds it of to the nearest state
# space value. For RPY values it rounds it off to the nearest 0.01 value and for XYZ pos values it rounds it off to
# the nearest 0.001 value
    '''
    def create_indices(self, state_values):

        # The one at the end signifies its a unit value
        rot_r1, rot_p1, rot_y1, pos_x1, pos_y1, pos_z1 = np.meshgrid(self.unit_linspace, self.unit_linspace,
                                                                     self.unit_linspace, self.unit_linspace,
                                                                     self.unit_linspace, self.unit_linspace)
        mapped_rot_r = zip(rot_r1, state_values[0])
        mapped_rot_p = zip(rot_p1, state_values[1])
        mapped_rot_y = zip(rot_y1, state_values[2])
        mapped_pos_x = zip(pos_x1, state_values[3])
        mapped_pos_y = zip(pos_y1, state_values[4])
        mapped_pos_z = zip(pos_z1, state_values[5])

        return mapped_rot_r, mapped_rot_p, mapped_rot_y, mapped_pos_x, mapped_pos_y, mapped_pos_z

    
        To round off to the nearest 0.005 decimal, do math.floor(x*200)/200, where 200 is basically 1/0.05
        end_pos_x = math.floor(state_values[3]*200/float(200))
        end_pos_y = math.floor(state_values[4]*200/float(200))
        end_pos_z = math.floor(state_values[5]*200/float(200))
        index_end_pos_x = (end_pos_x + 0.5)*10
        index_end_pos_y = (end_pos_y + 0.5)*10
        index_end_pos_z = (end_pos_z + 0.5)*10
        index_rot_par_r = (rot_par_r+math.pi)*50/math.pi + 0.5
        index_rot_par_p = (rot_par_p+math.pi)*50/math.pi + 0.5
        index_rot_par_y = (rot_par_y+math.pi)*50/math.pi + 0.5
    '''

    def get_next_state(self, curr_state, action):
        # curr_state = np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z])
        # Since the state value is normalized by dividing with 2*pi, so multiply with 2*pi and add action
        # Then divide final result by 2*pi to normalize the data again
        next_state_val = (curr_state*2*np.pi + action)/(2*np.pi)
        return next_state_val

    def get_indices(self, state_values):

        index_rot_par_r = state_values[0]
        index_rot_par_p = state_values[1]
        index_rot_par_y = state_values[2]
        index_end_pos_x = state_values[3]
        index_end_pos_y = state_values[4]
        index_end_pos_z = state_values[5]
        # (z*10 + 0.09)/float(0.006)
        # The limit values for the model being created are the following:
        # It is based on the data created by the user, the minimum and maximum values of the data
        # model_pos_x_val (-0.009, -0.003)
        # model_pos_y_val (0.003, 0.007)
        # model_pos_z_val (-0.014, -0.008)
        # model_rot_r_val (-0.5, 0.5)
        # model_rot_p_val (-0.234, -0.155)
        # model_rot_y_val (0.28, 0.443)
        index_rot_par_r = (index_rot_par_r * 10 + 5) / float(1)
        index_rot_par_p = (index_rot_par_p * 10 + 5) / float(0.079)
        index_rot_par_y = (index_rot_par_y * 10 + 5) / float(0.163)
        index_end_pos_x = (index_end_pos_x * 10 + 0.09) / float(0.006)
        index_end_pos_y = (index_end_pos_y * 10 + 0.09) / float(0.006)
        index_end_pos_z = (index_end_pos_z * 10 + 0.09) / float(0.006)
        index_values = np.array([index_rot_par_r.astype(int), index_rot_par_p.astype(int), index_rot_par_y.astype(int),
                                 index_end_pos_x.astype(int), index_end_pos_y.astype(int), index_end_pos_z.astype(int)])

        return index_values

    def main_loop(self, action):
        # new_state_rot_par_r, new_state_rot_par_p, new_state_rot_par_y, new_state_end_pos_x, new_state_end_pos_y, \
        # new_state_end_pos_z = self.get_next_state(self.state_values, action)
        new_state_values = self.get_next_state(self.state_values, action)

        new_index_values = self.get_indices(new_state_values)

        q = self.r[self.model_index_values] + 0.9*self.v[new_index_values]
        p = np.exp(0.75*q)
        return q, p

    def initial_loop(self):
        # print "Calculating q..."
        q = self.r[self.model_index_values]
        # print "Calculating p..."
        p = np.exp(0.75*q)
        return q, p


    def get_policy(self, weights, n_iter, n_time):

        print 'Creating state space...'
        self.model_state_values = np.meshgrid(self.model_rot_r_val, self.model_rot_p_val, self.model_rot_y_val,
                                              self.model_pos_x_val, self.model_pos_y_val, self.model_pos_z_val,
                                              sparse=True)
        print 'State space created.'

        # Creating action set
        # The rotation values have accuracy of 0.01 and position values have 0.001 accuracy
        for rot_r in [-0.01, 0, 0.01]:
            for rot_p in [-0.01, 0, 0.01]:
                for rot_y in [-0.01, 0, 0.01]:
                    for pos_x in [-0.001, 0, 0.001]:
                        for pos_y in [-0.001, 0, 0.001]:
                            for pos_z in [-0.001, 0, 0.001]:
                                self.action_set.append(np.array([rot_r, rot_p, rot_y, pos_x, pos_y, pos_z]))

        # Get the reward and feature values for all the model state values
        self.r, self.f = features.reward(self.model_state_values, weights)

        # Get the index value for each of the model state value
        self.model_index_values = self.get_indices(self.model_state_values)

        policy = []
        for iter in range(0, n_iter):
            action_value = []
            policy = []
            print "Policy Iteration:", iter
            # start_time = t.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                if iter == 0:
                    func = self.initial_loop
                else:
                    func = self.main_loop
                for q, p in executor.map(func, self.action_set):
                    action_value.append(q)
                    policy.append(p)

            print "Evaluating Policy..."
            policy = policy/sum(policy)
            self.v = sum(policy*action_value)
            # end_time = t.time()
            # print end_time-start_time

        # mu = np.empty([301,301,101,11])
        print "Final Policy evaluated."
        print "Calulating State Visitation Frequency..."
        mu = np.exp(-(float(self.model_state_values[0]))**2)*np.exp(-(float(self.model_state_values[1]))**2) * \
             np.exp(-(float(self.model_state_values[2]))**2)*np.exp(-(float(self.model_state_values[3]))**2) * \
             np.exp(-(float(self.model_state_values[4]))**2)*np.exp(-(float(self.model_state_values[5]))**2)
        mu_reshape = np.reshape(mu, [11*11*11*11*11*11, 1])
        mu = mu/sum(mu_reshape)
        mu_last = mu
        print "Initial State Frequency calculated..."
        for time in range(0, n_time):
            s = np.zeros([11, 11, 11, 11, 11, 11])
            for act_index, action in enumerate(self.action_set):
                new_state_values = self.get_next_state(self.model_state_values, action)

                new_index_values = self.get_indices(new_state_values)

                p = policy[act_index, self.model_index_values]
                s = s + p*mu_last[new_index_values]
            mu_last = s
            mu = mu + mu_last
        mu = mu/n_time
        state_visitation = mu_last*self.f
        print "State Visitation Frequency calculated."
        return np.sum(state_visitation.reshape(2, 11*11*11*11*11*11), axis=1), policy

