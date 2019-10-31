from environment import Environment
from read_write_joint_to_file import DataCollection
# obj_read_data = DataCollection(25, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv")
# x, y, z, d, e, fg = obj_read_data.data_parse_as_numpy_arr()
# print"array feature", x, y, z, d, e, fg
# obj = Environment(1)
# feature_matrix = obj.feature_matrix()
# n_states, d_states = feature_matrix.shape
# print "n states and d states is ", n_states, d_states
# print" Feature matrix example ", feature_matrix[0]

# obj.generate_trajectories_data()
# obj.edited_write_data_trajectories_file("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_data_2_trial.csv")

# obj2 = DataCollection(1, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_data_1_1000hz.csv")
# val = obj2.data_parse_as_numpy_arr()
from numpy import genfromtxt
my_data = genfromtxt("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_data_1_1000hz.csv", delimiter=',')
print "val is ", my_data
'''
# Moving dvrk arm with small values
import PyKDL
import dvrk
import numpy as np
# Create a Python proxy for PSM1, name must match ros namespace
arm = dvrk.psm('PSM1')

# You can home from Python
arm.home()
print(arm.get_current_position())
j = arm.get_desired_position()
goal = PyKDL.Frame()
pos = j.p
rpy = j.M.GetRPY()
print type(arm.get_desired_position().M.GetRPY())
for i in np.arange(0, 0.005, 0.0001):
    goal.M = PyKDL.Rotation.RPY(rpy[0]+i, rpy[1]+i, rpy[2]+i)
    # goal.M = arm.get_desired_position().M
    goal.p = PyKDL.Vector(pos[0], pos[1], pos[2]+i)
    arm.move(goal)
print"move completed"
'''