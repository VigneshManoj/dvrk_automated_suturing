import numpy as np
from sys import getsizeof
import matplotlib.pyplot as plt
import csv
import math
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
# print a
# for i in range(0, 5):
#     c = np.append(a, [2, 3, 4, 5, 7], axis=0)
#     # print a
#     print "reps"
# print c
# end_pos_y = np.array([[0.338]])
# x= np.round(end_pos_y.astype(float)**2,3)
# print x
# print round(float(x)**2, 3)
end_pos_x = np.linspace(0, 2, 3, dtype='float16')
end_pos_y = np.linspace(0, 2, 3, dtype='float16')
end_pos_z = np.linspace(0, 2, 3, dtype='float16')

rot_par_r = np.linspace(-0.014, -0.008, 6, dtype='float16')
rot_par_p = np.linspace(-0.014, -0.008, 6, dtype='float16')
# rot_par_r = np.linspace(0, 2, 3, dtype='float16')
# rot_par_p = np.linspace(0, 2, 3, dtype='float16')
rot_par_y = np.linspace(0, 2, 3, dtype='float16')
# end_pos_x = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_y = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_z = np.linspace(-0.015, 0.008, 11, dtype='float16')
#
# rot_par_r = np.linspace(-0.5, 0.5, 11, dtype='float16')
# rot_par_p = np.linspace(-0.5, 0.5, 11, dtype='float16')
# rot_par_y = np.linspace(-0.5, 0.5, 11, dtype='float16')
# x = np.linspace(0, 2, 3, dtype='float16')
# y = np.linspace(0, 2, 3, dtype='float16')
# xv, yv = np.meshgrid(x, y)  # make sparse output arrays
# print "xv is ", xv
# x = np.linspace(-0.014, -0.008, 11)
# print x
# xv, yv, zv, xv2, yv2, zv2 = np.meshgrid(x, y, z, x2, y2, z2)  # make sparse output arrays
state_rot_par_r, state_rot_par_p = \
         np.meshgrid(rot_par_r, rot_par_p)
# print state_end_pos_x
# a = np.zeros((10, 10, 10))
try:
    value_index = state_rot_par_p.index(-0.008)
except:
    value_index = -1
# print a[0][0]
print "value index ", value_index
print state_rot_par_r[0]
print state_rot_par_p
# state_rot_par_r, state_rot_par_p, state_rot_par_y, state_end_pos_x, state_end_pos_y, state_end_pos_z = np.meshgrid\
       #  (rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z)
# print state_rot_par_r
# plt.plot(x, y)
# plt.show()
# for i in np.arange(-0.015, 0.008, 0.001):
        # for j in np.arange(-0.015, 0.008, 0.001):

# print state_rot_par_r[0][0][0][0][0][1]
# print "zv", zv
# print "xv2", xv2.shape
# print "yv2", yv2
# print "zv2", zv2
# x1 = 0.525
# x1 = np.round(x1,2)
# y = np.round(y,2)
#
#
# x1[x1<=-1.5] = -1.5
# x1[x1 >= 1.5] = 1.5
#
# print x1