# Inspired from Phil Tabor's Youtube channel "Machine learning with Phil"
import numpy as np
import concurrent.futures
from robot_markov_model import RobotMarkovModel


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.03, 0.02], [0.025, 0.075], [-0.14, -0.09]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.lin_space_limits_x = np.linspace(-0.03, 0.02, self.grid_size, dtype='float32')
        self.lin_space_limits_y = np.linspace(0.025, 0.075, self.grid_size, dtype='float32')
        self.lin_space_limits_z = np.linspace(-0.14, -0.09, self.grid_size, dtype='float32')

        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        # Total Number of states defining the state of the robot
        self.n_params_for_state = 3
        # The terminal state value which is taken from the expert trajectory data
        self.terminal_state_val = terminal_state_val_from_trajectory
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size ** 3
        self.n_actions = 27
        self.gamma = discount
        self.rewards = np.zeros([self.n_states])

    # Creates the state space of the robot based on the values initialized for linspace by the user
    def create_state_space_model_func(self):
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits_x:
            for j_val in self.lin_space_limits_y:
                for k_val in self.lin_space_limits_z:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 4), round(j_val, 4), round(k_val, 4)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states

    # Creates the action space required for the robot. It is defined by the user beforehand itself
    def create_action_set_func(self):
        action_set = []
        for pos_x in [-0.005, 0, 0.005]:
            for pos_y in [-0.005, 0, 0.005]:
                for pos_z in [-0.005, 0, 0.005]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space

    # Function to convert 3D position coordinates into index values of the grid world
    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.03) / 0.005 * pow(self.grid_size, 2)) + \
                    abs((state_val[1] - 0.025) / 0.005 * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.14) / 0.005)
        return int(round(index_val))

    # Function to check if the reached state is the terminal state
    def is_terminal_state(self, state):

        # because terminal state is being given in array value and needs to convert to index value
        terminal_state_val_index = self.get_state_val_index(self.terminal_state_val)
        if int(state) == int(terminal_state_val_index):
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
            # reward = 1 if rewards[int(state)] > 1 else 0
            # It has not yet reached the terminal state
            return False

    # Check if the agent is moving outside the grid
    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        sum_feat = np.zeros(len(self.states))
        for i, ele in enumerate(self.states.values()):
            sum_feat[i] = np.all(np.equal(ele, new_state))
        if np.sum(sum_feat) == 0:
            return True
        # if trying to wrap around the grid, also the reason for the for x in _ is because old_state is a list
        # elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in
        #                                                        new_state) == self.grid_size - 1:
        #     return True
        # elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in
        #                                                                         new_state) == 0:
        #     return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False

    # Function to reset the agent positon
    def reset(self):
        self.pos = np.random.randint(0, len(self.states))
        return self.pos

    # Function which computes the next state S' from a state S when an action A is taken
    def step(self, curr_state, action):
        resulting_state = []
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_params_for_state):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value and
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            reward = self.rewards[int(resulting_state_index)]
            return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], -1, self.is_terminal_state(resulting_state_index), None

    # Function to randomly sample an action (for exploration)
    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))


# Finds the action with the highest Q value
def max_action(Q, state_val, action_values):
    q_values = np.array([Q[state_val, a] for a in action_values])
    action = np.argmax(np.array([q_values]))
    return action_values[action]


# Function to implement Q learning
def q_learning(env_obj, alpha, gamma, epsilon, max_num_steps):

    Q = {}
    # Set the number of episodes
    number_episodes = 5000
    total_rewards = np.zeros(number_episodes)
    # Initialize the Q(s,a) values
    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0
    # Run it as many times as required number of episodes
    for i in range(number_episodes):
        done = False
        ep_rewards = 0
        # Reset the environment at the start of each episode
        observation = env_obj.reset()
        # Count to ensure the episode is reset if the agent is stuck
        count = 0
        # Until the agent reaches the goal state
        while not done:
            # Sample a random value to exploit or explore
            rand = np.random.random()
            # print "----------------------------------------------------------------------------"
            # Select the best action or explore a new action
            action = max_action(Q, observation, env_obj.action_space.keys()) if rand < (1 - epsilon) \
                else env_obj.action_space_sample()
            # Find the resulting state
            observation_, reward, done, info = env_obj.step(observation, action)
            # Add up rewards of an episode
            ep_rewards += reward
            # Resulting state index value in grid world
            next_observation_index = env_obj.get_state_val_index(observation_)
            # visited_states.append(next_observation_index)
            # Q learning implementation
            action_ = max_action(Q, next_observation_index, env_obj.action_space.keys())
            Q[observation, action] = Q[observation, action] + \
                                     alpha * (reward + gamma * Q[next_observation_index, action_] -
                                              Q[observation, action])
            observation = next_observation_index
            # Increment step
            count += 1
            # End episode if maximum number of steps is crossed
            if count > max_num_steps:
                break
        # Update epsilon value to reduce exploration as the number of episodes increases
        if epsilon - 2 / number_episodes > 0:
            epsilon -= 2 / number_episodes
        else:
            epsilon = 0
        # Add each episode reward
        total_rewards[i] = ep_rewards

    return Q, total_rewards


if __name__ == '__main__':
    # Define cartesian goal point
    goal = np.array([0.005, 0.055, -0.125])
    # Create the Object of class which creates the state space and action space
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.01, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print("State space created is ", states)
    # Initialize policy and rewards
    policy = np.zeros(len(states))
    rewards = np.zeros(len(states))
    index = env_obj.get_state_val_index(goal)
    # To store the user trajectories
    mdp_obj = RobotMarkovModel()
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    # Find the index values of all states visited by the user
    for i in range(len(trajectories[0])):
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    # Assign Sparse Reward
    # Terminal state gets higher rewards, visited states get +1 and all other states get 0 reward
    for _, ele in enumerate(index_vals):
        if ele == index:
            rewards[int(ele)] = 10
        else:
            rewards[int(ele)] = 1
    # Pass the rewards to Q learning function object
    env_obj.rewards = rewards
    # Call Q learning function
    Q, total_rew = q_learning(env_obj, alpha=0.1, gamma=0.01, epsilon=1, max_num_steps=100)

    # Compute the policy based on Q values of state space
    for s in states:
        Q_for_state = [Q[int(s), int(a)] for a in action]
        policy[int(s)] = np.argmax(Q_for_state)
    print(" policy is ", policy)
    # Store the policy
    folder_dir = "/home/vignesh/Desktop/individual_trials/version4/data1/"
    np.save(folder_dir + 'policy', policy)






