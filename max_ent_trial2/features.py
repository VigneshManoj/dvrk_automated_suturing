import numpy as np
import math


# Calculates reward function
def reward(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z, weights):

    features = [feature1, feature2]
    r = 0
    f = []
    for n in range(0, len(features)):
        f.append(features[n](rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z))
        r = r+weights[0, n]*f[n]
# Created the feature function assuming everything has importance, so therefore added each parameter value
    return r, f


# Created feature set1 which basically takes the exponential of sum of individually squared value
def feature1(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
    f1 = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
    return f1


# Created feature set2 which basically takes the exponential of sum of individually squared value divided by
# the variance value
def feature2(rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z):
    f2 = np.exp(-(rot_par_r**2 + rot_par_p**2 + rot_par_y**2 + end_pos_x**2 + end_pos_y**2 + end_pos_z**2)/0.1**2)
    # print f2
    return f2

