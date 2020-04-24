import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sys import getsizeof
import matplotlib.pyplot as plt
import csv
import math
from pprint import pprint
# rot_par_r = np.linspace(-math.pi, math.pi, 101, dtype='float64')
# rot_par_s = np.linspace(-math.pi, math.pi, 1001, dtype='float64')
# csv = np.genfromtxt ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data.csv" , delimiter=",")
# # print rot_par_r
# # print rot_par_s
# print csv[:, 0:6:12]
# import pandas as pd
# df = pd.read_csv("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data.csv")
# # print df
# X = df.iloc[:, 0:6]
# # print "x is ", type(X)
# a = np.zeros([1, 6])
# model_rot_r_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
# model_rot_p_val = np.linspace(-0.5, 0, 11, dtype='float16')
# model_rot_y_val = np.linspace(0, 0.5, 11, dtype='float16')
# model_pos_x_val = np.linspace(0.00, -0.005, 11, dtype='float16')
# model_pos_y_val = np.linspace(0.00, 0.005, 11, dtype='float16')
# model_pos_z_val = np.linspace(-0.05, -0.01, 11, dtype='float16')
# print round(float(x)**2, 3)
# model_rot_r_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
# model_rot_p_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
# model_rot_y_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
# model_pos_x_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
# model_pos_y_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
# model_pos_z_val = np.linspace(-0.009, -0.003, 11, dtype='float16')

# end_pos_x = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_y = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_z = np.linspace(-0.015, 0.008, 11, dtype='float16')
#
# x1 = np.linspace(-0.5, 0.5, 11)
# y1 = np.linspace(0., 0.5, 11)
# z1 = np.linspace(-0.009, -0.003, 11)
#
# x, y, z = np.meshgrid(x1, y1, z1, sparse=False)
# # print x
# # print y
# integer_values = [0, 1, 2, 3, 4, 5]
# xt = (x*10 + 5)/1
# # y = (y*10 + 15)/3
# yt = (y*10 )/0.5
# zt = (z*10 + 0.09)/float(0.006)
# # z = (z*10)/float(-0.003)
# model_state_values0, model_state_values1, model_state_values2, model_state_values3, model_state_values4, model_state_values5  = np.meshgrid(model_rot_r_val, model_rot_p_val, model_rot_y_val,
#                                               model_pos_x_val, model_pos_y_val, model_pos_z_val, sparse=True)
# index_rot_par_r = model_state_values0
# index_rot_par_p = model_state_values1
# index_rot_par_y = model_state_values2
# index_end_pos_x = model_state_values3
# index_end_pos_y = model_state_values4
# index_end_pos_z = model_state_values5
# index_rot_par_r = (index_rot_par_r * 10 + 5) / float(1)
# index_rot_par_p = (index_rot_par_p * 10 + 5) / float(1)
# index_rot_par_y = (index_rot_par_y * 10 + 5) / float(1)
# index_end_pos_x = (index_end_pos_x * 10 + 0.09) / float(0.006)
# index_end_pos_y = (index_end_pos_y * 10 + 0.09) / float(0.006)
# index_end_pos_z = (index_end_pos_z * 10 + 0.09) / float(0.006)
# print model_state_values0.shape
# print index_end_pos_z.astype(int)

# print xt[5][0][0] # 5
# print x[5][5][0]
# print "z is", z
'''
# X = (['X' for key, i in enumerate(x1)], [key for key, i in enumerate(x1)], [i for i in x1])
X = {key: i for key, i in enumerate(x1)}
Y = {key: i for key, i in enumerate(y1)}
Z = {key: i for key, i in enumerate(x1)}
TH = {key: i for key, i in enumerate(y1)}
PH = {key: i for key, i in enumerate(x1)}
SC = {key: i for key, i in enumerate(y1)}

AX = ['X', 'Y', 'Z', 'TH', 'PH', 'SC']
Keys = [i for i in range(11)]


def ret_id(x,y,z,th,ph,sc):
    return (X[x], Y[y], Z[z], TH[th], PH[ph], SC[sc])

States_map = {
    'X': X, 'Y': Y, 'Z': Z, 'TH': TH, 'PH': PH, 'SC': SC
}
pprint(States_map)

for ax in AX:
    for key in Keys:
        k = States_map[ax]
        print (ax, States_map[ax], key, k[key])
'''
state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z = [2, 2, 2, 2, 2, 2]

arr = np.array([state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z])
mu = np.exp(-(state_rot_par_r)**2)*np.exp(-(state_rot_par_p )**2) * \
             np.exp(-(state_rot_par_y )**2)*np.exp(-(state_end_pos_x )**2) * \
             np.exp(-(state_end_pos_y )**2)*np.exp(-(state_end_pos_z )**2)
mu2 = np.exp(-arr**2)
print mu
print mu2
# print States_map['X'][5]

# print str(m) + ', ' + str(n)
# x = [[i for i in range(n)] for j in range(m)]
# y = (y*10 + 15)/3
# pprint(x)
# print y
# print zi
# pped[, 1.5]

