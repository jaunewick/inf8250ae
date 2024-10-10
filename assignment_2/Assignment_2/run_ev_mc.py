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

all_sa_values, all_returns = [], []
behaviour_policy = lambda state: np.full(9, 1./9)

for i in range(5):
    sa_values, returns = algo.ev_mc_off_policy_control(env, behaviour_policy, epsilon=0.01, num_episodes=1500, discount=0.99)
    all_sa_values.append(sa_values)
    all_returns.append(returns)

with open('data/ev_mc_off_policy_sa_values.npy', 'wb') as f:
    np.save(f, all_sa_values)
with open('data/ev_mc_off_policy_returns.npy', 'wb') as g:
    np.save(g, all_returns)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(all_returns)



