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
        # Deterministic or stochastic transition environment
        self.trans_prob = 1
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size**3
        self.n_actions = 27
        self.gamma = discount
        self.temp_state = 0
        self.visited_states_index = []
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

    # Creates the action space required for the robot. It is desfined by the user beforehand itself
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

    # Computes the reward for the agent
    def compute_reward(self, state):
        if self.is_terminal_state(state):
            reward = 10
        else:
            reward = -1

        return reward

    # Function to check if the reached state is the terminal state
    def is_terminal_state(self, state):
        # because terminal state is being given in array value and needs to convert to index value
        terminal_state_val_index = self.get_state_val_index(self.terminal_state_val)
        if int(state) == int(terminal_state_val_index):
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
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
        # elif (x % self.grid_size for x in old_state) == 0 and
        # (y % self.grid_size for y in new_state) == self.grid_size - 1:
        #     return True
        # elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and
        # (y % self.grid_size for y in new_state) == 0:
        #     return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False

    # Function which computes the next state S' from a state S when an action A is taken
    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_params_for_state):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value and
        # number of features based on the features provided
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            reward = self.compute_reward(int(resulting_state_index))
            return resulting_state_index, reward
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return curr_state, -1

    def calc_value_for_state(self, a):
        resulting_state_index, reward = self.step(self.temp_state, a)
        value = reward + self.gamma * V[resulting_state_index]
        return value, a

    def value_iteration(self, V, policy, error):
        converged = False
        i = 0
        action_range_value = range(0, self.n_actions)

        while not converged:
            delta = 0
            for state in self.states.keys():
                # print "state is ", state
                i += 1
                old_qvalue = V[int(state)]
                # print old_qvalue
                current_qvalue = np.zeros([self.n_states])
                self.temp_state = state
                for q, a in self.map(self.calc_value_for_state, action_range_value):
                    current_qvalue[state] = q
                best_qvalue = np.where(current_qvalue == current_qvalue.max())[0]
                highest_qvalue_state = np.random.choice(best_qvalue)
                V[state] = current_qvalue[highest_qvalue_state]
                delta = max(delta, np.abs(old_qvalue - V[state]))
                converged = True if delta < error else False

        for state in self.states.keys():
            new_qvalues = []
            actions = []
            i += 1
            for action in self.action_space.keys():
                resulting_state_index, reward = self.step(state, action)
                new_qvalues.append(reward + self.gamma * V[resulting_state_index])
                actions.append(action)
            new_qvalues = np.array(new_qvalues)
            best_action_index = np.where(new_qvalues == new_qvalues.max())[0]
            best_action = actions[best_action_index[0]]
            policy[state] = best_action
        print('completed ', i, ' number of sweeps of the state space')
        return V, policy


if __name__ == '__main__':
    goal = np.array([0.005, 0.055, -0.125])
    env_obj = RobotStateUtils(11, 0.1, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index_val = env_obj.get_state_val_index(goal)
    print "index val is ", index_val
    mdp_obj = RobotMarkovModel()
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    # rewards = np.zeros(len(states))
    # for j in range(len(rewards)):
    #     rewards[j] = 0
    # for _, ele in enumerate(index_vals):
    #     if ele == index_val:
    #         rewards[int(ele)] = 100
    #     else:
    #         rewards[int(ele)] = 1
    for i in range(len(trajectories[0])):
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    env_obj.visited_states_index = index_vals
    # Initialize Value function
    V = {}
    for state in env_obj.states:
        V[state] = 0
    # Perform value iteration (currently set to 5 times)
    policy = {}
    for state in env_obj.states:
        policy[state] = [key for key in env_obj.action_space]
    # Set the error value limit to 0.001 for stopping value iteration loop (between two iterations of value iteration)
    for i in range(5):
        V, policy = env_obj.value_iteration(V, policy, 0.001)
    print(policy)

    policy_arr = np.zeros(len(states))
    for j, ele in policy.items():
        policy_arr[j] = ele
