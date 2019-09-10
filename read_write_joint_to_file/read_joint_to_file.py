# !/usr/bin/env python

import dvrk
import numpy as np
import rospy
import math


class DataCollection:
    # To make sure the headings exist in the csv file and is only written once
    file_heading_exits = 0

    def __init__(self, loop_rate):
        self.loop_rate = loop_rate

    def read_joint_write_file(self):
        rospy.init_node('read_write_data_dvrk')
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace
        p = dvrk.psm('PSM1')
        p.home()
        # p.move_joint_one(0.05, 0)
        # Location of file storage
        data_file_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv"
        csv = open(data_file_dir, "a")
        # For writing the heading to the csv file
        if self.file_heading_exits == 0:
            column_title = "Joint 1, Joint 2, Joint 3, Joint 4, Joint 5, Joint 6\n"
            csv.write(column_title)
            self.file_heading_exits = 1
        # While loop to read and write data continuously
        while not rospy.is_shutdown():
            # Read joint positions of dvrk arm
            current_pos = p.get_current_joint_position()
            print("\nThe current joint position is ", current_pos, "\n")
            row_data = str(current_pos[0])
            # Write the data to file
            # For writing each joint position value
            for i in range(1, 6):
                row_data += "," + str(current_pos[i])
                print("\n I value is ", i, "\n")
                # New set of data starts in a new line
                if i == 5:
                    row_data = row_data + "\n"
                    print("reached inside the next line")
            # Finally write the data in row_data to the csv file
            csv.write(row_data)
            # Decides at what rate the data is written to the csv file
            rate.sleep()



