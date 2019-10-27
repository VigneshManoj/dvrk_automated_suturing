
import PyKDL
import dvrk
import numpy as np
# Create a Python proxy for PSM1, name must match ros namespace
arm = dvrk.psm('PSM1')

# You can home from Python
arm.home()
print(arm.get_current_position())
j = arm.get_desired_position()
goal = PyKDL.Frame()
pos = j.p
rpy = j.M.GetRPY()
print type(arm.get_desired_position().M.GetRPY())
for i in np.arange(0, 0.005, 0.0001):
    goal.M = PyKDL.Rotation.RPY(rpy[0]+i, rpy[1]+i, rpy[2]+i)
    # goal.M = arm.get_desired_position().M
    goal.p = PyKDL.Vector(pos[0], pos[1], pos[2]+i)
    arm.move(goal)
print"move completed"