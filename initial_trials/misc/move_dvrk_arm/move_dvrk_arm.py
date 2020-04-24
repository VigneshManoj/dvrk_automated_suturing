import dvrk
import numpy as np
import rospy
import math
import PyKDL
from misc.jigsaw_data_parse import jigsaw_data_parser

class MoveDVRKArm:

    def __init__(self, loop_rate):
        self.loop_rate = loop_rate
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace

    def jigsaw_move_dvrk_arm(self):
        # rospy.init_node('move_dvrk_arm')
        p = dvrk.psm('PSM1')
        p.home()
        # Create a Python proxy for PSM1, name must match ros namespace
        # p.move(0.001, 0)
        tip_pos_x, tip_pos_y, tip_pos_z, frame_val = jigsaw_data_parser()
        for i in range(0, 4):
            p.move(PyKDL.Vector(0.5*tip_pos_x[i], 0.5*tip_pos_y[i], 0.5*tip_pos_z[i]))
            print(tip_pos_x[i], tip_pos_y[i], tip_pos_z[i])

    def move_dvrk_arm_cartesian(self, x, y, z):
        p.move(PyKDL.Vector(x, y, z))




