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
        obj_read_data = DataCollection(1000, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data_1_1000hz.csv")
        arr_feature_vec = obj_read_data.data_parse_as_numpy_arr()
        # Return the feature vector from collected data for a specific state
        return arr_feature_vec.iloc[state_val]

    def feature_matrix(self):

        features = []
        # It starts at 2 because first two rows are names and none type, for now have randomly chosen 300, later need to make it modular
        for n in range(2, 5000):
            # Create temporary variable to normalize all the feature vectors (angles/2*pi and pos/norm_val and vel/largest_vel)
            ### Write code to find the largest velocity value in the collected data and use that value to normalize velocity function ###
            temp_f = self.feature_vector(n)
            # Finding the normalizing value for the 3 position vectors
            norm_dist = sqrt(pow(round(float(temp_f[3]), 3), 2) + pow(round(float(temp_f[4]), 3), 2) + pow(round(float(temp_f[5]), 3), 2))
            f = [round(float(temp_f[0]), 3)/(2*pi), round(float(temp_f[1]), 3)/(2*pi), round(float(temp_f[2]), 3)/(2*pi), round(float(temp_f[3]), 3)/norm_dist, round(float(temp_f[4]), 3)/norm_dist, round(float(temp_f[5]), 3)/norm_dist]
            # Add all the normalize feature values to a list
            features.append(f)
        return np.array(features)

    def generate_trajectories_data(self):
        obj_read_data = DataCollection(1000, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data_1_1000hz.csv")
        state_values = obj_read_data.data_parse_as_numpy_arr()
        actions_col = []
        actions_row = []
        # Reads each row starting from third (first and second row is unwanted data)
        for i in range(2, 5000):
            # Reads all the 6 columns of the data file
            for j in range(0, 6):
                # Reads two consecutive row values for a specific column and subtracts
                # It rounds up 3 decimal places for angles and 4 decimal places for positions
                if j < 3:
                    act = round(round(float(state_values.iloc[i + 1][j]), 4) - round(float(state_values.iloc[i][j]), 4), 4)
                else:
                    act = round(round(float(state_values.iloc[i + 1][j]), 4) - round(float(state_values.iloc[i][j]), 4), 4)
                # Creates an array of the actions for a state
                actions_col.append(act)
            # Creates an array of the actions for all the states
            actions_row.append(actions_col)
            # Resets the list to zero
            actions_col = []
        # print "Action is ", actions_row[0], actions_row[1]
        return actions_row

    def write_data_trajectories_file(self, file_dir):
        csv = open(file_dir, "a")
        row_data = self.generate_trajectories_data()
        for i in range(2, 5000):
            temp_str = self.feature_vector(i)
            str_row_data = str(round(float(temp_str.iloc[0]), 4)) + "," + str(round(float(temp_str.iloc[1]), 4)) + "," + str(round(float(temp_str.iloc[2]), 4)) + "," + str(round(float(temp_str.iloc[3]), 4)) + "," + str(round(float(temp_str.iloc[4]), 4)) + "," + str(round(float(temp_str.iloc[5]), 4))
            print("str data is ", str_row_data)
            # Writes the action file into the csv file
            for j in range(0, 6):
                str_row_data += "," + str(row_data[i - 2][j])
            csv.write(str_row_data + '\n')

