# !/usr/bin/env python

import dvrk
import PyKDL
import numpy as np
from read_write_joint_to_file import DataCollection
from move_dvrk_arm import MoveDVRKArm
# # # Create a Python proxy for PSM1, name must match ros namespace
# p = dvrk.psm('PSM1')
# p.home()
# p.move(PyKDL.Vector(0.020315, 0.03192, -0.014969))
# p.move_joint_one(-0.05, 2) # move 3rd joint

# # obj = MoveDVRKArm(0.05, 0, 0, 0, 0)
# # print("reached here")
# # #p.dmove(PyKDL.Vector(0.0, 0.02, 0.0)) # 5 cm in Y direction
# # # p.move(PyKDL.Vector(0.0, 0.0, 0.0))
# # obj.move_dvrk_arm_single_joint()
#
#
# # !/usr/bin/env python
# #
# import dvrk
# # import numpy
# # import rospy
# # import math
# # import PyKDL
# # i = 0.25
# # j, k = 0, 0
# if __name__ == '__main__':
# #     rospy.init_node('move_davinci1')
# #     # Create a Python proxy for PSM1, name must match ros namespace
#      p = dvrk.psm('PSM1')
#      p.home()
# #     # r = PyKDL.Rotation()
# #     rate = rospy.Rate(500.0)
# #     while not rospy.is_shutdown():
# #         # print "reaching here ", math.pi * i
# #         p.dmove_joint_one(i, 0)
# #
# #         if j <= 3:
# #             print "j val is ", j
# #             j = j + 1
# #             i = 0.25
# #             k = 0
# #         else:
# #             k = k + 1
# #             print "k val is ", k
# #             i = -0.25
# #             if k > 3:
# #                 j = 0
# #
#      current_pos = p.get_current_joint_position()
#      print("current pos is ", current_pos[0])
# #         current_vel = p.get_current_joint_velocity()
# #         current_effort = p.get_current_joint_effort()
# #         print "Current pos is ", current_pos
# #         print "Current vel is ", current_vel
# #         print "Current effort is ", current_effort
# #         rate.sleep()
# #
# # #example of writing data to a CSV
# #
# # dic = {"John": "john@example.com", "Mary": "mary@example.com"} #dictionary
# #
# # download_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv" #where you want the file to be downloaded to
# #
# # csv = open(download_dir, "w")
# # #"w" indicates that you're writing strings to the file
# #
# # columnTitleRow = "name, email\n"
# # csv.write(columnTitleRow)
# #
# # for key in dic.keys():
# # 	name = key
# # 	email = dic[key]
# # 	row = name + "," + email + "\n"
# # 	csv.write(row)
# #
# #
# #
# count = 0
# obj = DataCollection(10,"/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt")
# #obj.dvrk_data_write_to_file()
# each_line = obj.read_from_txt_file()
# x = np.array(each_line)
# print x[0][1]
# print("each line data of 3", each_line[2])
# print("length ", len(each_line))
# print("single element ", each_line[0][5])
# x = np.loadtxt("/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt")
# print "x is ",x[0]
# print "y is ", x[0][39]
# each_line = [i.split('\t')[0] for i in each_line]
# print"each line", each_line
# x = np.loadtxt("/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt")
# print("x is ", x[0])
# print("y is ", x[0][38])
#
# pos_ele_x = np.empty(0)
# pos_ele_y = np.empty(0)
# pos_ele_z = np.empty(0)
# for i in range(0, 4316):
#     pos_ele_x = np.append(pos_ele_x, x[i][38])
#     pos_ele_y = np.append(pos_ele_y, x[i][39])
#     pos_ele_z = np.append(pos_ele_z, x[i][40])
# print "x pos is ", pos_ele_x[0:3]

obj1 = MoveDVRKArm(10)
obj1.move_dvrk_arm()