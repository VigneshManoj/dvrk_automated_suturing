# !/usr/bin/env python

import dvrk
import numpy as np
import rospy
import math
import csv

# from read_write_joint_to_file import MoveDVRKArm
file_heading_exits = 0

if __name__ == '__main__':
    rospy.init_node('move_davinci')
    rate = rospy.Rate(500.0)
    # Create a Python proxy for PSM1, name must match ros namespace
    p = dvrk.psm('PSM1')
    p.home()
    # p.move_joint_one(0.05, 0)
    data_file_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv"  # where you want the file to be downloaded to
    csv = open(data_file_dir, "a")

    current_pos = p.get_current_joint_position()
    print("\nThe current joint position is ", current_pos, "\n")
    if file_heading_exits == 0:
        columnTitleRow = "Joint 1, Joint 2, Joint 3, Joint 4, Joint 5, Joint 6\n"
        csv.write(columnTitleRow)
        file_heading_exits = 1
    for i in range(1, 6):
        # row =  current_pos[0] + "," + current_pos[1] + "," + current_pos[2] + "," + current_pos[3] + "," + current_pos[4] + "," + current_pos[5]"\n"
        row = str(current_pos[0])
        row += "," + str(current_pos[i])
        if i == 6:
            row += "\n"
        csv.write(row)