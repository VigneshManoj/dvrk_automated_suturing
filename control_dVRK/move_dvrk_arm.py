import rospy
import dvrk
import PyKDL
import numpy as np
from robot_state_utils import RobotStateUtils


if __name__ == '__main__':
    # Initialize a rospy init node for dVRK
    rospy.init_node('move_dvrk_arm')
    p = dvrk.psm('PSM2')
    p.home()
    rospy_rate = 10
    # Define the starting cartesian position
    starting_point = np.array([-0.008, 0.053, -0.134])
    # Load the policy to follow
    policy_folder_dir = '/home/vignesh/PycharmProjects/dvrk_automated_suturing/learnt_policies/'
    policy = np.load(policy_folder_dir + 'best_policy.npy')
    # Define cartesian goal point
    goal_point = np.array([0.005, 0.055, -0.125])
    # Create the Object of class which creates the state space and action space
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.8, goal_point)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    # Get the starting and goal position index values in the grid world
    start_index = env_obj.get_state_val_index(starting_point)
    goal_index = env_obj.get_state_val_index(goal_point)
    done = False
    current_state_idx = start_index
    # Repeat until the agent reaches the goal state
    limit_val = 0
    while not done:
        # Find the action to take based on the policy learnt
        action_idx = policy[start_index]
        # Find the resulting state
        next_state = states[current_state_idx] + action[action_idx]
        # Command the dVRK to move to new position
        p.move(PyKDL.Vector(float(next_state), float(next_state), float(next_state)))
        print("PSM arm new position is ", next_state[0], next_state[1], next_state[2])
        next_state_idx = env_obj.get_state_val_index(next_state)
        current_state_idx = next_state_idx
        # Check if the resulting state is the goal state
        if int(current_state_idx) == int(goal_index):
            done = True
        limit_val += 1
        # To ensure the loop ends if the agent gets stuck at some local minima
        if limit_val > 100:
            print("Agent is stuck between consecutive states, exiting!")
            exit()


