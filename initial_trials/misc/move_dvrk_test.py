import dvrk
# Create a Python proxy for PSM1, name must match ros namespace
p = dvrk.psm('ECM')

# You can home from Python
p.home()
p.get_current_joint_position()
# p.dmove_joint_one(-0.05, 2) # move 3rd joint
p.move_joint_one(0.01, 2)