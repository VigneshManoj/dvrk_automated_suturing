import numpy as np
import pandas as pd
import rospy
import dvrk
import PyKDL
from read_write_joint_to_file import DataCollection
# from move_dvrk_arm import MoveDVRKArm


if __name__ == '__main__':
    file_dir = "/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/832953.csv"
    rospy.init_node('move_dvrk_arm')
    p = dvrk.psm('PSM2')
    p.home()
    rospy_rate = 10
    read_obj = DataCollection(rospy_rate, file_dir)
    # move_dvrk_obj = MoveDVRKArm(rospy_rate)
    df = read_obj.data_parse_df_numpy_arr()
    # print df[1:3],  "\nhi\n", df[1:3][0], df[1:3][2], "\nhi\n", df.shape[0]
    for i in range(1, df.shape[0]):
        p.move(PyKDL.Vector(float(df[0][i]), float(df[2][i]), float(df[4][i])))
        print "Movement is ", df[0][i], df[2][i], df[4][i]