import numpy as np
from robot_state_utils import RobotStateUtils


if __name__ == '__main__':

    folder_dir = '/home/vignesh/PycharmProjects/dvrk_automated_suturing/learnt_policies/'
    # Best policy learnt by Q learning
    policy = np.load(folder_dir + 'best_policy.npy')
    # Second best policy learnt by Q learning
    # policy = np.load(folder_dir + 'second_best_policy.npy')
    # Third best policy learnt by Q learning
    # policy = np.load(folder_dir + 'third_best_policy.npy')
    # Miscellaneous good policies learnt by Q learning
    # policy = np.load(folder_dir + 'misc_policy1.npy')
    # policy = np.load(folder_dir + 'misc_policy2.npy')

    # Define the starting 3D coordinate position of agent
    starting_point = np.array([-0.025, 0.05, -0.135])
    # Define the goal position (3D coordinate) of agent
    goal_point = np.array([0.005, 0.055, -0.125])
    # Initialize an array of visited states by agent
    visited_states = starting_point
    # Create the Object of class which creates the state space and action space
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.1, goal_point)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    # Get the starting and goal position index values in the grid world
    start_index = env_obj.get_state_val_index(starting_point)
    goal_index = env_obj.get_state_val_index(goal_point)
    done = False
    current_state_idx = start_index
    print "policy is ", policy[current_state_idx]
    # Repeat until the agent reaches the goal state
    limit_val = 0
    while not done:
        next_state = []
        # Find the action to take based on the policy learnt
        action_idx = policy[current_state_idx]
        print "action to be taken ", action[action_idx]
        # Find the resulting state
        for i in range(0, 3):
            next_state.append(round(states[current_state_idx][i] + action[int(action_idx)][i], 4))
        print "State is ", next_state[0], next_state[1], next_state[2]
        # Store the visited states by the agent
        visited_states = np.append(visited_states, np.array(next_state), axis=0)
        current_state_idx = env_obj.get_state_val_index(np.array(next_state))
        # Check if the resulting state is the goal state
        if int(current_state_idx) == int(goal_index):
            done = True
            np.save(folder_dir + 'visited_states_output', visited_states)
        limit_val += 1
        # To ensure the loop ends if the agent gets stuck at some local minima
        if limit_val > 250:
            print("Agent is stuck between consecutive states, exiting!")
            exit()



