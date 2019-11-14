from itertools import product

import numpy as np
import numpy.random as rn

from value_iteration import find_policy as val_find_policy
from value_iteration import value as val_value

def irl(feature_matrix, n_actions, discount, trajectories, epochs, learning_rate):


    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = rn.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        # print("i: {}".format(i))
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += learning_rate * grad

    return feature_matrix.dot(alpha).reshape((n_states,))

def find_svf(n_states, trajectories):

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount, trajectories):

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = val_find_policy(n_states, n_actions, r, discount)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j])

    return expected_svf.sum(axis=1)

def softmax(x1, x2):

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount):

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(V[k] for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([ V[k] for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions,
    reward, discount, p_start_state, optimal_value, true_reward):

    policy = val_find_policy(n_states, n_actions,
        reward, discount)
    value = val_value(policy.argmax(axis=1), n_states,
       true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd

