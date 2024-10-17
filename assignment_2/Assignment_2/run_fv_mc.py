import random

import matplotlib.pyplot as plt
import numpy as np

import algorithms as algo
import utils as utl
from environment import RaceTrack

# Set seed
seed = 0
np.random.seed(seed)
random.seed(seed)
render_mode = None

env = RaceTrack(track_map='c', render_mode=render_mode)

# all_sa_values, all_returns = [], []
# for i in range(5):
#     sa_values, returns = algo.fv_mc_control(env, epsilon=0.01, num_episodes=1500, discount=0.99)
#     all_sa_values.append(sa_values)
#     all_returns.append(returns)

# with open('data/fv_mc_sa_values.npy', 'wb') as f:
#     np.save(f, all_sa_values)
# with open('data/fv_mc_returns.npy', 'wb') as g:
#     np.save(g, all_returns)

# plt.figure(figsize=(15, 7))
# plt.grid()
# utl.plot_many(all_returns)

ZERO_GREEDY = 0
HALF_GREEDY = 0.5

with open('data/fv_mc_sa_values.npy', 'rb') as f:
    all_sa_values = np.load(f, allow_pickle=True)

prev_last_sa_values = all_sa_values[-1]
eps_greedy_policy = utl.make_eps_greedy_policy(prev_last_sa_values, epsilon=HALF_GREEDY)
eps_greedy_returns = []

for _ in range(5):
    states, actions, rewards = utl.generate_episode(eps_greedy_policy, env)
    eps_greedy_returns.append(sum(rewards))

print(f"Accumulated returns obtained in each episode run using {HALF_GREEDY}-greedy policy:", eps_greedy_returns)

average_returns = np.mean(eps_greedy_returns)
print(f"Average returns obtained using {HALF_GREEDY}-greedy policy: {average_returns}")

