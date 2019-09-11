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
        # p.move_joint_one(0.05, 0)
        # Location of file storage
        data_file_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv"
        csv = open(data_file_dir, "a")
        # For writing the heading to the csv file
        if self.file_heading_exits == 0:
            p.home()
            column_title = "Pos Joint 1, Pos Joint 2, Pos Joint 3, Pos Joint 4, Pos Joint 5, Pos Joint 6, " \
                           "Vel Joint 1, Vel Joint 2, Vel Joint 3, Vel Joint 4, Vel Joint 5, Vel Joint 6 \n"
            csv.write(column_title)
            self.file_heading_exits = 1
        # While loop to read and write data continuously
        while not rospy.is_shutdown():
            # Read joint positions of dvrk arm
            current_pos = p.get_current_joint_position()
            # Read joint velocities of dvrk arm
            current_vel = p.get_current_joint_velocity()
            # print("\nThe current joint position is ", current_pos, "\n")
            # print("\nThe current joint position is ", current_vel, "\n")

            # Initialize the row data with joint 0 position
            row_data = str(current_pos[0])
            # Write the data to file
            # For writing each joint position value
            for i in range(1, 6):
                row_data += "," + str(current_pos[i])
                # print("\n I value is ", i, "\n")
                if i == 5:
                    # For writing each joint velocity value
                    row_data += "," + str(current_vel[0])
                    for j in range(1, 6):
                        row_data += "," + str(current_vel[j])
                        # New set of data starts in a new line
                        if j == 5:
                            row_data = row_data + "\n"
            # Finally write the data in row_data to the csv file
            csv.write(row_data)
            # Decides at what rate the data is written to the csv file
            rate.sleep()


    def read_from_txt_file(self):
        # Path of the folder containing the data
        txt_file_dir = "/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt"
        # Open the file
        open_txt_file = open(txt_file_dir, "r")
        # Reads all the lines and stores it and returns the data
        concat_lines = open_txt_file.readlines()
        return concat_lines


