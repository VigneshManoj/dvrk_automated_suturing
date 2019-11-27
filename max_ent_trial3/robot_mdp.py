import numpy as np
import math

class RobotMarkovModel:
    def __init__(self):

    # Calculates reward function
    def reward(self, state_values, weights):

        features = [feature1, feature2]
        r = 0
        f = []
        for n in range(0, len(features)):
            f.append(features[n](state_values[0], state_values[1], state_values[2],
                                 state_values[3], state_values[4], state_values[5]))
            r = r+weights[0, n]*f[n]
    # Created the feature function assuming everything has importance, so therefore added each parameter value
        return r, f


    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
        feature = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
        return feature


    # Created feature set2 which basically takes the exponential of sum of individually squared value divided by
    # the variance value
    def feature2(self, rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
        f2 = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2)
        # print f2
        return f2
