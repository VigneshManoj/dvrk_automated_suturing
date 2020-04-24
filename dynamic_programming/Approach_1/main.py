import numpy as np
from robot_state_utils import RobotStateUtils
from numpy import savetxt
from robot_markov_model import RobotMarkovModel


if __name__ == '__main__':
    # Location to store the computed policy
    file_name = "/home/vignesh/Desktop/individual_trials/version4/data1/policy_grid11.txt"
    # term_state = np.random.randint(0, grid_size ** 3)]
    goal = np.array([0.005, 0.055, -0.125])
    # Create objects of classes
    env_obj = RobotStateUtils(11, 0.01, goal)
    mdp_obj = RobotMarkovModel()
    # Store the expert trajectories
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    for i in range(len(trajectories[0])):
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    # print index_vals
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    rewards = np.zeros(len(states))
    index = env_obj.get_state_val_index(goal)
    for _, ele in enumerate(index_vals):
        if ele == index:
            rewards[int(ele)] = 10
        else:
            rewards[int(ele)] = 1

    policy = env_obj.value_iteration(rewards)
    file_open = open(file_name, 'a')
    savetxt(file_open, policy, delimiter=',', fmt="%10.5f", newline=", ")
    file_open.write("\n \n \n \n")
    file_open.close()



