import random

import matplotlib.pyplot as plt
import numpy as np

import utils as utl
from environment import RaceTrack
from td_algos import QLearningAgent, Sarsa, td_control

# Set seed
seed = 0
np.random.seed(seed)
random.seed(seed)
render_mode = None

num_episodes = 2000
env = RaceTrack(track_map='c', render_mode=render_mode)

info = {"env": env,
        "step_size": 0.1,
        "epsilon": 0.05,
        "discount": 0.99,
        'seed': 0
        }

returns = []
agents = []

for i in range(5):
        all_returns, agent = td_control(env, agent_class=Sarsa, info=info, num_episodes=num_episodes)
        returns.append(all_returns)
        agents.append(agent)

with open('data/td_sarsa_returns.npy', 'wb') as f:
        np.save(f, returns)
with open('data/td_sarsa_agents.npy', 'wb') as g:
        np.save(g, agents)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(returns)