import numpy as np
import time
from sys import getsizeof
action_set = []
leng = 0
start_time = time.time()
for rot_par_r in [-0.01, 0, 0.01]:
    for rot_par_p in [-0.01, 0, 0.01]:
        for rot_par_y in [-0.01, 0, 0.01]:
            for end_pos_x in [-0.001, 0, 0.001]:
                for end_pos_y in [-0.001, 0, 0.001]:
                    for end_pos_z in [-0.001, 0, 0.001]:
                            action_set.append(np.array(
                                [rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z]))
                            # print action_set
                            leng += 1


print "length is ", leng, len(action_set), getsizeof(action_set)
print "time taken ", (time.time() - start_time)