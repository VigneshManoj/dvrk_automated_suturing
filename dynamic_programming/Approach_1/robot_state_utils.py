import numpy as np
import concurrent.futures


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount_factor, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.03, 0.02], [0.025, 0.075], [-0.14, -0.09]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.lin_space_limits_x = np.linspace(-0.03, 0.02, self.grid_size, dtype='float32')
        self.lin_space_limits_y = np.linspace(0.025, 0.075, self.grid_size, dtype='float32')
        self.lin_space_limits_z = np.linspace(-0.14, -0.09, self.grid_size, dtype='float32')

        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        self.possible_actions = [i for i in range(27)]
        # Total Number of states defining the state of the robot
        self.n_params_for_state = 3
        # The terminal state value which is taken from the expert trajectory data
        self.terminal_state_val = terminal_state_val_from_trajectory
        # Deterministic or stochastic transition environment
        self.trans_prob = 1
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size**3
        self.n_actions = 27
        self.gamma = discount_factor
        self.pos = 0
        # Initialize rewards, transition probability
        self.rewards = []
        self.P_a = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.int32)
        self.values_tmp = np.zeros([self.n_states])

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
        index_value = abs((state_val[0] + 0.03) / 0.005 * pow(self.grid_size, 2)) + \
                    abs((state_val[1] - 0.025) / 0.005 * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.14) / 0.005)
        return int(round(index_value))

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
        if new_state not in self.states.values():
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

    # Computes the reward for the agent
    def reward_func(self, state):
        if self.is_terminal_state(state):
            reward = 10
        else:
            reward = -1

        return reward

    # Function to reset the agent positon
    def reset(self):
        self.pos = np.random.randint(0, len(self.states))
        return self.pos

    # Function which computes the next state S' from a state S when an action A is taken
    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_params_for_state):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        reward = self.reward_func(resulting_state_index)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], reward, self.is_terminal_state(curr_state), None

    # Function to randomly sample an action
    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))

    # Function to compute the transition state probabilities
    def get_transition_states_and_probs(self, curr_state, action):

        if self.is_terminal_state(curr_state):
            return [(curr_state, 1)]
        resulting_state = []
        if self.trans_prob == 1:
            for i in range(0, self.n_params_for_state):
                resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))
            resulting_state_index = self.get_state_val_index(resulting_state)

            if not self.off_grid_move(resulting_state, self.states[curr_state]):
                # return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
                return [(resulting_state_index, 1)]
            else:
                # if the state is invalid, stay in the current state
                return [(curr_state, 1)]

    # Create transition matrix
    def get_transition_mat_deterministic(self):

        self.n_actions = len(self.action_space)
        for si in range(self.n_states):
            for a in range(self.n_actions):
                probabilities = self.get_transition_states_and_probs(si, a)
                for next_pos, prob in probabilities:
                    sj = int(next_pos)
                    prob = int(prob)
                    self.P_a[si, a, sj] = prob
        return self.P_a

    # Computes the value of a state
    def calc_value_for_state(self, s):
        value = max([sum([self.P_a[s, a, s1] * (self.rewards[s] + self.gamma * self.values_tmp[s1]) for s1 in range(self.n_states)])
                     for a in range(self.n_actions)])
        return value, s

    # Function that computes the policy using value iteration technique
    def value_iteration(self, rewards, error=1):
        # Initialize the value function
        values = np.zeros([self.n_states])
        states_range_value = range(0, self.n_states)
        # print "states range value is ", states_range_value
        self.rewards = rewards
        # estimate values
        while True:
            # Temporary copy to check find the difference between new value function calculated & current value function
            # to ensure improvement in value
            self.values_tmp = values.copy()
            # t_value = TicToc()
            # t_value.tic()
            for q, s in self.map(self.calc_value_for_state, states_range_value):
                values[s] = q
                # print "\nvalues is ", values[s]
            # t_value.toc('Value function section took')
                # print "values ", values[s]
            if max([abs(values[s] - self.values_tmp[s]) for s in range(self.n_states)]) < error:
                break
        # generate deterministic policy
        policy = np.zeros([self.n_states])
        for s in range(self.n_states):
            policy[s] = np.argmax([sum([self.P_a[s, a, s1] * (self.rewards[s] + self.gamma * values[s1])
                                        for s1 in range(self.n_states)])
                                   for a in range(self.n_actions)])

        return policy

    # Function to compute state visitation frequency which is used for Maximum Entropy IRL
    def compute_state_visitation_frequency(self, trajectories, optimal_policy):
        n_trajectories = len(trajectories)
        total_states = len(trajectories[0])
        d_states = len(trajectories[0][0])
        T = total_states
        mu = np.zeros([self.n_states, T])
        # print "mu is ", mu
        for trajectory in trajectories:
            # print "trajectory is ", trajectory
            # To get the values of the trajectory in the state space created
            trajectory_index = self.get_state_val_index(trajectory[0])
            # int is added because the index returned is float and the index value for array has to be integer
            mu[int(trajectory_index), 0] += 1
        mu[:, 0] = mu[:, 0] / n_trajectories

        for s in range(self.n_states):
            for t in range(T - 1):
                # Computes the mu value for each state once the optimal action is taken
                mu[s, t + 1] = sum([mu[pre_s, t] * self.P_a[pre_s, int(optimal_policy[pre_s]), s]
                                    for pre_s in range(self.n_states)])
        p = np.sum(mu, 1)
        return p


if __name__ == '__main__':
    example_goal = np.array([[0.005, 0.025, -0.09]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, 0.9, example_goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index_val = env_obj.get_state_val_index(states[53])
    print "index val is ", index_val




















