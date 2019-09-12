import dvrk
import numpy as np
import rospy
import math
import PyKDL
from jigsaw_data_parse import jigsaw_data_parser

class MoveDVRKArm:

    def __init__(self, loop_rate):
        self.loop_rate = loop_rate

    def move_dvrk_arm(self):
        rospy.init_node('move_dvrk_arm')
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace
        p = dvrk.psm('PSM1')
        p.home()
        tip_pos_x, tip_pos_y, tip_pos_z = jigsaw_data_parser()
        for i in range(0, 4):
            p.move(PyKDL.Vector(tip_pos_x[i], tip_pos_y[i], tip_pos_z[i]))

