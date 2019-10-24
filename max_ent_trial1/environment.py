# !/usr/bin/env python

import numpy as np
import rospy
from math import pi, sqrt, pow
from read_write_joint_to_file import DataCollection

class Environment:

    def __init__(self, discount):

        self.actions = [-0.01, 0, 0.01, -0.001, 0.001]
        self.n_actions = len(self.actions)
        # Observable state features: angle 3 values, pos 3 values and velocity 1 value
        self.obs_state_features = 7
        self.n_states = pow(self.n_actions, self.obs_state_features)
        self.discount = discount

        '''
        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        '''

    def feature_vector(self, state_value):
        obj_read_data = DataCollection(25, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv")
        arr_feature_vec = obj_read_data.read_from_txt_file()
        return arr_feature_vec[state_value]

    def feature_matrix(self):

        features = []
        for n in range(self.n_states):
            temp_f = self.feature_vector(n)
            norm_dist = sqrt(pow(temp_f[4], 2) + pow(temp_f[5], 2) + pow(temp_f[6], 2))
            f = (temp_f[0]/(2*pi), temp_f[1]/(2*pi), temp_f[0]/(2*pi), temp_f[0]/norm_dist, temp_f[0]/norm_dist, temp_f[0]/norm_dist, temp_f[0]/2)
            features.append(f)
        return np.array(features)


