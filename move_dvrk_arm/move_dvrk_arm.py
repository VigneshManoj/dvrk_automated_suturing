import dvrk
import numpy as np
import rospy
import math


class MoveDVRKArm:

    def __init__(self, loop_rate):
        self.loop_rate = loop_rate

    def move_dvrk_arm(self):
        rospy.init_node('move_dvrk_arm')
        rate = rospy.Rate(self.loop_rate)
        # Create a Python proxy for PSM1, name must match ros namespace
        p = dvrk.psm('PSM1')
        p.home()
