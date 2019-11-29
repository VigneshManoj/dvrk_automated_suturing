import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
model_limits_rot_r_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
model_limits_rot_p_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
model_limits_rot_y_val = np.linspace(-0.5, 0.5, 11, dtype='float16')
model_limits_pos_x_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
model_limits_pos_y_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
model_limits_pos_z_val = np.linspace(-0.009, -0.003, 11, dtype='float16')

# model_rot_par_r, model_rot_par_p, model_rot_par_y, model_end_pos_x, model_end_pos_y, \
# model_end_pos_z = np.meshgrid(model_limits_rot_r_val,
#                                    model_limits_rot_p_val,
#                                    model_limits_rot_y_val,
#                                    model_limits_pos_x_val,
#                                    model_limits_pos_y_val,
#                                    model_limits_pos_z_val)
state_space = []
for r in np.arange(-0.5, 0.5, 11):
    for p in np.arange(-0.5, 0.5, 11):
        for y in np.arange(-0.5, 0.5, 11):
            for x in np.arange(-0.009, -0.003, 11):
                for yy in np.arange(-0.009, -0.003, 11):
                    for z in np.arange(-0.009, -0.003, 11):
                        state_space.append(np.array([r, p, y, x, yy, z]))

print state_space
