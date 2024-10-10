import random

import matplotlib.pyplot as plt
import numpy as np

import utils as utl
from utils import DistributionPolicy, State
from environment import RaceTrack
from algorithms import is_mc_estimate_with_ratios
from tqdm import trange

STATE_TO_TRACK = ((11, 5, 0, 0))    # You can modify this to see the evolution of V(s) for different states


def ois_mc_pred(env: RaceTrack, b_policy: DistributionPolicy, target_policy: DistributionPolicy, num_episodes: int, discount: float = 0.99) -> dict[State, list[float]]:
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    value_function_trace = {}
    b_policy_action_sampling = utl.convert_to_sampling_policy(b_policy)

    for episode in trange(num_episodes):
        states, actions, rewards = utl.generate_episode(b_policy_action_sampling, env)
        state_action_returns_and_ratios = is_mc_estimate_with_ratios(states, actions, rewards, target_policy, b_policy, discount)
        for state_action, returns_and_ratios in state_action_returns_and_ratios.items():
            if not state_action in all_state_action_values.keys():
                all_state_action_values[state_action] = []

            for ret, is_ratio in returns_and_ratios:
                all_state_action_values[state_action].append(ret * is_ratio)

        for state_action, values in all_state_action_values.items():
            state_action_values[state_action] = np.mean(values)

        for state, values in utl.qs_from_q(state_action_values).items():
            if state not in value_function_trace.keys():
                value_function_trace[state] = [np.mean(values)]
            else:
                value_function_trace[state].append(np.mean(values))

    return value_function_trace


# Set seed
seed = 0
np.random.seed(seed)
random.seed(seed)
render_mode = None

env = RaceTrack(track_map='c', render_mode=render_mode)
b_policy = lambda x: np.full(9, 1./9)

with open('data/fv_mc_sa_values.npy', 'rb') as f:
    all_sa_values = np.load(f, allow_pickle=True)

with open('data/fv_mc_returns.npy', 'rb') as g:
    all_returns = np.load(g, allow_pickle=True)

all_tracked_values = []

for i in range(5):
    target_policy = utl.make_eps_greedy_policy_distribution(all_sa_values[i], epsilon=0.2)
    value_function_trace = ois_mc_pred(env, b_policy, target_policy, 1500, discount=0.99)
    all_tracked_values.append(value_function_trace[STATE_TO_TRACK])

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(all_tracked_values, label=f"Value of state {STATE_TO_TRACK}")
