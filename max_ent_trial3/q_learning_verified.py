import numpy as np
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, terminal_state_val):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.lin_space_limits = np.linspace(-0.5, 0.5, self.grid_size, dtype='float32')
        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        self.possible_actions = [i for i in range(27)]
        # Total Number of states defining the state of the robot
        self.n_states = 3
        # self.current_pos = 1000
        self.terminal_state_val = terminal_state_val
        self.trans_prob = 1

    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 1), round(j_val, 1), round(k_val, 1)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        action_set = []
        for pos_x in [-0.5, 0, 0.5]:
            for pos_y in [-0.5, 0, 0.5]:
                for pos_z in [-0.5, 0, 0.5]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.5) * pow(self.grid_size, 2)) + abs((state_val[1] + 0.5) * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.5))
        return round(index_val*(self.grid_size-1))

    def is_terminal_state(self, state):

        # because terminal state is being given in index val
        if state == self.terminal_state_val:
        # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
            # It has not yet reached the terminal state
            return False

    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        if new_state not in self.states.values():
            return True
        # if trying to wrap around the grid, also the reason for the for x in _ is because old_state is a list
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in new_state) == 0:
            return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False


    def reset(self):
        # self.pos = np.random.randint(0, len(self.states))
        self.pos = 19
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        return self.pos

    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_states):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value, features value and
        # number of features based on the features provided
        # reward, features_arr, len_features = self.reward_func(resulting_state[0],
        #                                                       resulting_state[1],
        #                                                       resulting_state[2], self.weights)
        # print "reward is ", reward
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            return resulting_state, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], self.is_terminal_state(curr_state), None

    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))



def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    action = np.argmax(q_values)
    # print "---max action function action ", action
    # print "max q value ", q_values[action]
    return action_values[action]

# def q_learning(env_obj, alpha, gamma, epsilon):
def q_learning(env_obj, reward, alpha, gamma):

    # env_obj = RobotStateUtils(11, weights)
    # states = env_obj.create_state_space_model_func()
    # action = env_obj.create_action_set_func()
    # print "State space created is ", states
    Q = {}
    num_games = 50000
    total_rewards = np.zeros(num_games)
    epsilon = 0.2
    policy = {}
    state_trajectories = {}
    # Default value
    most_reward_index = 0
    sum_state_trajectory = 0
    expected_svf = np.zeros(len(env_obj.states))
    # print "obj state ", env_obj.states.keys()
    # print "obj action ", env_obj.action_space.keys()

    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0

    for i in range(num_games):
        if i % 10000 == 0:
            print('-------------starting game-------------- ', i)
        done = False
        ep_rewards = 0
        episode_policy = []
        state_trajectory = []
        observation = env_obj.reset()
        # observation = 0

        while not done:
            rand = np.random.random()
            # print "random val is ", rand
            # print "----------------------------------------------------------------------------"
            # print "Starting state val inside loop ", observation
            # print "action val inside loop", env_obj.action_space.keys()
            action = max_action(Q, observation, env_obj.action_space.keys()) if rand < (1 - epsilon) \
                else env_obj.action_space_sample()
            observation_, done, info = env_obj.step(observation, action)
            next_observation_index = env_obj.get_state_val_index(observation_)
            # print "reward is ", reward
            # print "next obs index is ", next_observation_index
            # print "reward at obs index ", reward[next_observation_index]
            ep_rewards += reward[int(next_observation_index)]
            # print "Next obs index", next_observation_index
            action_ = max_action(Q, next_observation_index, env_obj.action_space.keys())
             #print "current action val is ", action
            # print "next action val is ", action_
            # print "reward is ", reward

            Q[observation, action] = Q[observation, action] + \
                                     alpha * (reward[int(next_observation_index)] + gamma * Q[next_observation_index, action_] -
                                              Q[observation, action])
            # print "Q value in loop", Q[observation, action]
            episode_policy.append(np.exp(Q[observation, action]))
            # misc_val = Q[observation, action]
            # print "misc val1 ", Q[observation, action]
            # print "misc val2 ", alpha * (reward + gamma * Q[next_observation_index, action_] -
                                                                       # Q[observation, action])
            observation = next_observation_index
            # print "state value after assigning to new state", observation
            # state_trajectory.append(env_obj.states[observation])
        if epsilon - 2 / num_games > 0:
            epsilon -= 2 / num_games
        else:
            epsilon = 0
        total_rewards[i] = ep_rewards
        # policy[i] = episode_policy
        # most_reward_index = np.argmax(total_rewards)
        # policy_dict = {i: episode_policy}
        # policy.update(policy_dict)
        # state_dict = {i: state_trajectory}
        # state_trajectories.update(state_dict)
        # sum_state_trajectory = env_obj.sum_of_features(state_trajectories[most_reward_index])
    # expected_svf = env_obj.compute_state_visitation_freq(state_trajectories, policy[most_reward_index])
    return Q
    # return policy[most_reward_index], sum_state_trajectory, expected_svf

def optimal_policy_func(states, action, env_obj, weights, learning_rate, discount):
    Q = q_learning(env_obj, weights, learning_rate, discount)
    policy = np.zeros(len(states))
    for s in states:
        Q_for_state = [Q[int(s), int(a)] for a in action]
        # print "Q for each state is ", Q_for_state
        # print "state  ", s
        # policy[int(s)] = np.max(Q[int(s), int(a)] for a in action)
        policy[int(s)] = np.argmax(Q_for_state)
    # print " policy is ", policy

    return policy


if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    grid_size = 3
    feat_map = np.eye(grid_size**3)
    terminal_state = 16
    env_obj = RobotStateUtils(grid_size, terminal_state_val=terminal_state)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    # weights = np.array([[1, 1, 1]])
    # index = env_obj.get_state_val_index([0.5, -0.5, 0])
    # print "index is ", index
    # term_state = np.random.randint(0, grid_size ** 3)]

    print "State space created is ", states
    policy = np.zeros(len(states))
    # print "states is ", states[0], states[18]
    print "actions are ", action
    # r1 = np.array([0.93729132, 6.46284183, 0.59751722, 0.02687185, 8.29158771, 0.40334012,
    #                0.59266189, 6.86164796, 0.78016216, -0.22017214, 4.74269386, -0.60966791,
    #                0.15136383, 0.3804086, 0.20897725, 0.41426822, 6.3323203, 0.64039452,
    #                0.1334194, 2.89370163, 0.77631985, 0.16878807, 0.48677897, 0.67278622,
    #                0.03074749, 0.17259921, 0.38397193])
    r1 = np.array([0.07417828, 0.39015304, 0.07168588, 0.03540926, 0.51775156, 0.07111255,
                   0.04956109, 0.3892607,  0.01406589, 0.00224518, 0.08890297, 0.03781566,
                   0.0325048,  0.03470176, 0.04052829, 0.06753339, 0.38320178, 0.01707578,
                   0.06366697, 0.27192602, 0.01847448, 0.06519564, 0.05922704, 0.04938876,
                   0.06428576, 0.38818405, 0.0170321])
    # print "rewards is ", reward
    Q = q_learning(env_obj, reward=r1, alpha=0.1, gamma=0.8)
    # print "Q is ", Q
    # print "Q shape is ", len(Q)
    # print "Q values are ", Q.values()
    # az = [Q[0, int(a)] for a in action]
    # print "az is ", az
    for s in states:
        Q_for_state = [Q[int(s), int(a)] for a in action]
        # print "Q for each state is ", Q_for_state
        # print "state  ", s
        # policy[int(s)] = np.max(Q[int(s), int(a)] for a in action)
        policy[int(s)] = np.argmax(Q_for_state)
    print " policy is ", policy

    final_answer_1 = action[int(policy[19])]
    final_answer_2 = action[int(policy[10])]
    final_answer_3 = action[int(policy[1])]
    final_answer_4 = action[int(policy[4])]
    final_answer_5 = action[int(policy[7])]
    final_answer_6 = action[int(policy[16])]
    print "The actions 1 to be taken at state ", states[19], "is :", final_answer_1
    print "The actions 2 to be taken at state ", states[10], "is :", final_answer_2
    print "The actions 3 to be taken at state ", states[1], "is :", final_answer_3
    print "The actions 4 to be taken at state ", states[4], "is :", final_answer_4
    print "The actions 5 to be taken at state ", states[7], "is :", final_answer_5
    print "The actions 6 to be taken at state ", states[16], "is :", final_answer_6


# First answer
#  policy is  [ 17.  16.  15.  14.   0.  15.  11.  10.  12.  17.   7.   3.   2.   1.   0.
#    2.   0.   0.   7.   8.   6.  11.  10.   0.   5.   0.   0.]
# Second answer
#  policy is  [ 14.  16.  15.  11.   0.  10.  14.  10.   9.   4.   4.   3.   5.   1.   0.
#    2.   0.   0.   5.   6.   3.   2.   3.   0.   5.   0.   1.]




