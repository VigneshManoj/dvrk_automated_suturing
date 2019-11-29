import numpy as np
import numba as nb
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_angle = [[-0.5, 0.5], [-0.234, -0.155], [0.28, 0.443]]
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 6 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.model_limits_pos_x_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_limits_pos_y_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        self.model_limits_pos_z_val = np.linspace(-0.009, -0.003, 11, dtype='float16')
        # There are 6 parameters defining a state value of the robot, RPY and XYZ
        self.n_states = 3
        # The created model state values
        self.model_end_pos_x = []
        self.model_end_pos_y = []
        self.model_end_pos_z = []
        self.action_set = []
        # The created model values index positions
        self.model_index_pos_x = []
        self.model_index_pos_y = []
        self.model_index_pos_z = []
        # The next state values index positions and position values of the new state
        self.model_new_index_pos_x = []
        self.model_new_index_pos_y = []
        self.model_new_index_pos_z = []
        self.model_new_end_pos_x = []
        self.model_new_end_pos_y = []
        self.model_new_end_pos_z = []
        self.rewards = []
        self.features = []
        self.state_action_value = []
        self.discount = 0

    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        print "Creating State space "
        self.model_end_pos_x, self.model_end_pos_y, self.model_end_pos_z = np.meshgrid(self.model_limits_pos_x_val,
                                                                                       self.model_limits_pos_y_val,
                                                                                       self.model_limits_pos_z_val)
        print "State space has been created"

        # return self.model_rot_par_r, self.model_rot_par_p, self.model_rot_par_y, \
        #       self.model_end_pos_x, self.model_end_pos_y, self.model_end_pos_z

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        for pos_x in [-0.001, 0, 0.001]:
            for pos_y in [-0.001, 0, 0.001]:
                for pos_z in [-0.001, 0, 0.001]:
                    self.action_set.append(np.array([pos_x, pos_y, pos_z]))
        # return self.action_set

    def get_next_state(self, state_val_1, state_val_2, state_val_3, action):
        # curr_state = np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z])
        # Since the state value is normalized by dividing with 2*pi, so multiply with 2*pi and add action
        # Then divide final result by 2*pi to normalize the data again
        next_state_val_1 = (state_val_1 * 2 * np.pi + action[0])/(2 * np.pi)
        next_state_val_2 = (state_val_2 * 2 * np.pi + action[1])/(2 * np.pi)
        next_state_val_3 = (state_val_3 * 2 * np.pi + action[2])/(2 * np.pi)

        return next_state_val_1, next_state_val_2, next_state_val_3

    def policy_iteration_func(self, action):
        # new_state_rot_par_r, new_state_rot_par_p, new_state_rot_par_y, new_state_end_pos_x, new_state_end_pos_y, \
        # new_state_end_pos_z = self.get_next_state(self.state_values, action)
        self.model_new_end_pos_x, self.model_new_end_pos_y, self.model_new_end_pos_z = \
            self.get_next_state(self.model_end_pos_x, self.model_end_pos_y, self.model_end_pos_z, action)

        self.model_new_index_pos_x, self.model_new_index_pos_y, self.model_new_index_pos_z = \
            self.get_indices(self.model_new_end_pos_x, self.model_new_end_pos_y, self.model_new_end_pos_z)

        q = self.rewards[self.model_index_pos_x, self.model_index_pos_y, self.model_index_pos_z] + \
            self.discount*self.state_action_value[self.model_new_index_pos_x, self.model_new_index_pos_y, self.model_new_index_pos_z]
        p = np.exp(q)
        return q, p

    def initialize_policy_iteration_func(self, action):
        # print
        q = self.rewards[self.model_index_pos_x, self.model_index_pos_y, self.model_index_pos_z]
        # print "Calculating q...", q.shape
        # print "q of value is ", q[0][3][8]
        # print "Calculating p..."
        p = np.exp(q)
        return q, p

    def calculate_optimal_policy_func(self, alpha, discount, n_policy_iter):
        # Creates an object for using the RobotMarkovModel class
        robot_mdp = RobotMarkovModel()
        self.create_state_space_model_func()
        self.create_action_set_func()
        n_actions = len(self.action_set)
        self.discount = discount
        self.model_index_pos_x, self.model_index_pos_y, self.model_index_pos_z = self.get_indices(self.model_end_pos_x,
                                                                                                  self.model_end_pos_y,
                                                                                                  self.model_end_pos_z)

        self.rewards, self.features = robot_mdp.reward_func(self.model_end_pos_x, self.model_end_pos_y,
                                                            self.model_end_pos_z, alpha)
        # print "rewards is ", self.rewards.shape
        # print "reward 0 ", self.rewards[0][3][8]
        # return 0

        ### Note: look into trying to make this a Numpy array
        policy = []
        for i in range(0, n_policy_iter):
            action_value = []
            policy = []
            print "Policy Iteration:", i
            # start_time = t.time()

                # Run it on the first try to initialize the value of function of all the states to a value
                # func = self.initialize_policy_iteration_func(self.model_index_pos_x, self.model_index_pos_y, self.model_index_pos_z)
            if i == 0:
                func = self.initialize_policy_iteration_func
            else:
                func = self.policy_iteration_func
            # The map function was inherited from threadpoolexecutor function which is why its being called as self.map
            for q, p in self.map(func, self.action_set):
                action_value.append(q)
                policy.append(p)

            # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            #     if i == 0:
            #         # Run it on the first try to initialize the value of function of all the states to a value
            #         # func = self.initialize_policy_iteration_func(self.model_index_pos_x, self.model_index_pos_y, self.model_index_pos_z)
            #         func = outside_policy_iteration_func(1)
            #
            #     # else:
            #         # Run it later to calculate the value function of all the states present
            #         # func = self.policy_iteration_func()
            #     # print "action space ", self.action_set
            #     for q, p in executor.map(func, self.action_set):
            #         action_value.append(q)
            #         policy.append(p)
            print "Evaluating Policy..."
            policy = policy / sum(policy)
            self.state_action_value = sum(policy * action_value)
            print "state action value ", self.state_action_value
        return policy

    def get_indices(self, model_end_pos_x, model_end_pos_y, model_end_pos_z):
        index_end_pos_x = model_end_pos_x
        index_end_pos_y = model_end_pos_y
        index_end_pos_z = model_end_pos_z
        # (z*10 + 0.09)/float(0.006)
        # The limit values for the model being created are the following:
        # It is based on the data created by the user, the minimum and maximum values of the data
        # model_pos_x_val (-0.009, -0.003)
        # model_pos_y_val (0.003, 0.007)
        # model_pos_z_val (-0.014, -0.008)
        index_end_pos_x = (index_end_pos_x * 10 + 0.09) / float(0.006)
        index_end_pos_y = (index_end_pos_y * 10 + 0.09) / float(0.006)
        index_end_pos_z = (index_end_pos_z * 10 + 0.09) / float(0.006)

        return index_end_pos_x.astype(int), index_end_pos_y.astype(int), index_end_pos_z.astype(int)

def main(alpha, discount, n_policy_iter):
    # Creates an object for using the RobotMarkovModel class
    utils = RobotStateUtils()
    policy = utils.calculate_optimal_policy_func(alpha, discount, n_policy_iter)

if __name__ == '__main__':
    # The different kind of trajectories present in the user study
    trajectory_length = 1
    weights = np.random.rand(1, 2)
    # The number of times policy iteration needs to be run
    n_policy_iter = 3
    discount = 0.9
    main(weights, discount, n_policy_iter)
