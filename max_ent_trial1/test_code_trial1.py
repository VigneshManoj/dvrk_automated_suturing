from environment import Environment
from read_write_joint_to_file import DataCollection
# obj_read_data = DataCollection(25, "/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/dvrk_joint_data.csv")
# x, y, z, d, e, fg = obj_read_data.data_parse_as_numpy_arr()
# print"array feature", x, y, z, d, e, fg
obj = Environment(1)
feature_matrix = obj.feature_matrix()
n_states, d_states = feature_matrix.shape
print "n states and d states is ", n_states, d_states
obj.generate_trajectories_data()
obj.write_data_trajectories_file("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_data.csv")