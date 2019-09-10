# # import dvrk
# # import PyKDL
# # from read_write_joint_to_file import MoveDVRKArm
# #
# # # Create a Python proxy for PSM1, name must match ros namespace
# # # p = dvrk.psm('PSM1')
# # # p.home()
# # # p.dmove_joint_one(-0.05, 2) # move 3rd joint
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
