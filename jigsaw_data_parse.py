# !/usr/bin/env python
import numpy as np


def jigsaw_data_parser():
    x = np.loadtxt("/home/aimlabx/Downloads/Thesis/JIGSAWS/Suturing/kinematics/AllGestures/Suturing_I001.txt")

    # Initialize empty numpy arrays to store the x, y, z positions of the tip
    pos_ele_x = np.empty(0)
    pos_ele_y = np.empty(0)
    pos_ele_z = np.empty(0)
    frame_val = np.empty(0)
    # The total number of data present in JIGSAW dataset
    for i in range(0, len(x)):
        # Create numpy array of x, y, z positions
        pos_ele_x = np.append(pos_ele_x, x[i][38])
        pos_ele_y = np.append(pos_ele_y, x[i][39])
        pos_ele_z = np.append(pos_ele_z, x[i][40])
        frame_val = np.append(frame_val, x[i][41:50])
    return pos_ele_x, pos_ele_y, pos_ele_z, frame_val


