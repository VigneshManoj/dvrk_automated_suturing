# !/usr/bin/env python

import numpy as np
import rospy
from math import pi, sqrt, pow
from read_write_joint_to_file import DataCollection


class Environment:

    def __init__(self, discount):
        # Possible number of actions
        self.actions = [-0.01, 0, 0.01, -0.001, 0.001]
        # Total number of actions possible
        self.n_actions = len(self.actions)
        # Observable state features: angle 3 values, pos 3 values and velocity 1 value
        self.obs_state_features = 7
        # Total number of state spaces possible = action^state_features
        self.n_states = pow(self.n_actions, self.obs_state_features)
        # User defined discount value
        self.discount = discount

        '''
        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        '''

    def feature_vector(self, state_val):
        # Read from the collected data
        obj_read_data = DataCollection(25, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv")
        arr_feature_vec = obj_read_data.data_parse_as_numpy_arr()
        # Return the feature vector from collected data for a specific state
        return arr_feature_vec.iloc[state_val]

    def feature_matrix(self):

        features = []
        for n in range(2, 300):
            # Create temporary variable to normalize all the feature vectors (angles/2*pi and pos/norm_val and vel/largest_vel)
            ### Write code to find the largest velocity value in the collected data and use that value to normalize velocity function ###
            temp_f = self.feature_vector(n)
            # Finding the normalizing value for the 3 position vectors
            norm_dist = sqrt(pow(float(temp_f[3]), 2) + pow(float(temp_f[4]), 2) + pow(float(temp_f[5]), 2))
            f = (float(temp_f[0])/(2*pi), float(temp_f[1])/(2*pi), float(temp_f[2])/(2*pi), float(temp_f[3])/norm_dist, float(temp_f[4])/norm_dist, float(temp_f[5])/norm_dist)
            # Add all the normalize feature values to a list
            features.append(f)
        return np.array(features)


