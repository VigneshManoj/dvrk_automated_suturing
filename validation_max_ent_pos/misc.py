import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

model_limits_pos_x_val = np.linspace(0, 0.01, 11, dtype='float16')
model_limits_pos_y_val = np.linspace( 0,0.01, 11, dtype='float16')
model_limits_pos_z_val = np.linspace( 0, 0.01,11, dtype='float16')

model_end_pos_x, model_end_pos_y, model_end_pos_z = np.meshgrid(model_limits_pos_x_val, model_limits_pos_y_val,
                                                                model_limits_pos_z_val)

# x = (model_end_pos_y*10 )/float(0.01)
# print x.astype(int)
print np.vstack((model_limits_pos_x_val, model_limits_pos_y_val, model_limits_pos_z_val))