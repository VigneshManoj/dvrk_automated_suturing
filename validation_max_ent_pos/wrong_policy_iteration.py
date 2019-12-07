"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn

class Gridworld():
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = []
        self.n_actions = len(self.actions)
        self.n_states = grid_size**3
        self.grid_size = grid_size
        self.discount = discount

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        for pos_x in [-0.001, 0, 0.001]:
            for pos_y in [-0.001, 0, 0.001]:
                for pos_z in [-0.001, 0, 0.001]:
                    self.actions.append((pos_x, pos_y, pos_z))

    def return_action_set(self):
        return self.actions


    def __str__(self):
        return "Gridworld({}, {})".format(self.grid_size, self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """
        point = (round(i % self.grid_size), round(i // self.grid_size), round((i-(i // self.grid_size))//self.grid_size))
        return point

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """
        int_val = p[0] + p[1]*self.grid_size + (p[2]*self.grid_size + p[1]*self.grid_size)
        # This is using the logic: Convert matrix to linear list. x, y*scale, z*scale+y*scale
        return int_val

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) + abs(i[2] - k[2]) <= 1

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """
        sx, sy, sz = self.int_to_point(state_int)
        # print sx, sy, sz
        if sx < self.grid_size and sy < self.grid_size and sz < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        if sz < self.grid_size-1:
            return 2
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy, sz = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy, random_start=True):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy, sz = rn.randint(self.grid_size), rn.randint(self.grid_size), rn.randint(self.grid_size)
                print "Randomly assigned states are ", sx, sy, sz

            else:
                sx, sy, sz = 0, 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                # Follow the given policy.
                print "states are ", sx, sy, sz
                print "Policy ", self.point_to_int((sx, sy, sz))
                print "Ended "
                action = self.actions[policy(self.point_to_int((sx, sy, sz)))]

                if (0 <= sx + action[0] < self.grid_size and 0 <= sy + action[1] < self.grid_size
                        and 0 <= sz + action[2] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                    next_sz = sz + action[2]
                else:
                    next_sx = sx
                    next_sy = sy
                    next_sz = sz

                state_int = self.point_to_int((sx, sy, sz))
                # print "action is ", action
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy, next_sz))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy
                sz = next_sz

            trajectories.append(trajectory)

        return np.array(trajectories)

def main(grid_size, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    trajectory_length = 3*3*grid_size
    gw = Gridworld(grid_size, discount)
    gw.create_action_set_func()
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy)
    # print trajectories

if __name__ == '__main__':
    main(3, 0.01, 20, 200, 0.01)
