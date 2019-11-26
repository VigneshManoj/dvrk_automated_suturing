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

rot_par_r = np.linspace(-0.014, -0.008, 3, dtype='float16')
rot_par_p = np.linspace(-0.014, -0.008, 3, dtype='float16')
# rot_par_r = np.linspace(0, 2, 3, dtype='float16')
# rot_par_p = np.linspace(0, 2, 3, dtype='float16')
rot_par_y = np.linspace(0, 2, 3, dtype='float16')
# end_pos_x = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_y = np.linspace(-0.015, 0.008, 11, dtype='float16')
# end_pos_z = np.linspace(-0.015, 0.008, 11, dtype='float16')
#
x1 = np.linspace(-0.5, 0.5, 11)
y1 = np.linspace(0., 0.5, 11)
z1 = np.linspace(-0.009, -0.003, 11)

x, y, z = np.meshgrid(x1, y1, z1, sparse=False)
# print x
# print y
integer_values = [0, 1, 2, 3, 4, 5]
xt = (x*10 + 5)/1
# y = (y*10 + 15)/3
yt = (y*10 )/0.5
zt = (z*10 + 0.09)/float(0.006)
# z = (z*10)/float(-0.003)

print zt
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
# print States_map['X'][5]

# print str(m) + ', ' + str(n)
# x = [[i for i in range(n)] for j in range(m)]
# y = (y*10 + 15)/3
# pprint(x)
# print y
# print zi
# pped[, 1.5]

