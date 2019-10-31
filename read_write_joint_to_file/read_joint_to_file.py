# !/usr/bin/env python

import dvrk
import numpy as np
import rospy
import math
import pandas as pd
from numpy import genfromtxt

class DataCollection:
    # To make sure the headings exist in the csv file and is only written once
    file_heading_exits = 0

    def __init__(self, loop_rate, file_dir):
        self.loop_rate = loop_rate
        self.file_dir = file_dir

    def dvrk_data_write_to_file_two_arms(self):
        # rospy.init_node('read_write_data_dvrk')
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace
        p = dvrk.psm('PSM1')
        # p.move_joint_one(0.05, 0)
        # Location of file storage
        # data_file_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv"
        csv = open(self.file_dir, "a")
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
        # txt_file_dir = "/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt"
        # Open the file
        open_txt_file = open(self.file_dir, "r")
        # Reads all the lines and stores it and returns the data
        concat_lines = open_txt_file.readlines()
        return concat_lines

    def read_jigsaw_dataset_file(self):
        # Open the file
        open_jigsaw_file = open(self.file_dir, "r")
        # Reads all the lines and stores it and returns the data
        read_lines = open_jigsaw_file.readlines()
        return read_lines


    def dvrk_data_write_to_file_single_arm(self):
        rospy.init_node('read_write_data_dvrk')
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace
        p = dvrk.psm('PSM1')
        # p.move_joint_one(0.05, 0)
        # Location of file storage
        # data_file_dir = "/home/aimlabx/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv"
        csv = open(self.file_dir, "a")
        # Initialization of rpy and pos
        current_rpy = np.zeros(3)
        current_pos = np.zeros(3)
        # For writing the heading to the csv file
        if self.file_heading_exits == 0:
            p.home()
            column_title = "Rot R, Rot P, Rot Y, Pos X, Pos Y, Pos Z \n"
            csv.write(column_title)
            self.file_heading_exits = 1
        # While loop to read and write data continuously
        while not rospy.is_shutdown():
            # Read joint positions and angles of dvrk arm
            current_pose = p.get_current_position()
            current_rpy = current_pose.M.GetRPY()
            current_pos = current_pose.p
            # print("\nThe current joint position is ", current_pos, "\n")
            # print("\nThe current joint position is ", current_vel, "\n")

            # Initialize the row data with joint 0 position
            row_data = str(current_rpy[0])
            # Write the data to file
            # For writing each end effector angle value
            for i in range(1, 3):
                row_data += "," + str(current_rpy[i])
                # print("\n I value is ", i, "\n")
                if i == 2:
                    # For writing each end effector position value
                    row_data += "," + str(current_pos[0])
                    print "Row data is ", row_data
                    for j in range(1, 3):
                        row_data += "," + str(current_pos[j])
                        # New set of data starts in a new line
                        if j == 2:
                            row_data = row_data + "\n"
            # Finally write the data in row_data to the csv file
            csv.write(row_data)
            # Decides at what rate the data is written to the csv file
            rate.sleep()

    def data_parse_df_numpy_arr(self):
        # Reads data using pandas and returns a pandas dataframe
        df = pd.read_csv(self.file_dir, sep=',', header=None)
        return df

    def data_parse_numpy(self):
        numpy_arr = genfromtxt(self.file_dir, delimiter=',')
        return numpy_arr


