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

sarsa_returns = []
sarsa_agents = []

for i in range(5):
        info['seed'] = i
        np.random.seed(info['seed'])
        random.seed(info['seed'])
        all_returns, agent = td_control(env, agent_class=Sarsa, info=info, num_episodes=num_episodes)
        sarsa_returns.append(all_returns)
        sarsa_agents.append(agent)

with open('data/td_sarsa_returns.npy', 'wb') as f:
        np.save(f, sarsa_returns)
with open('data/td_sarsa_agents.npy', 'wb') as g:
        np.save(g, sarsa_agents)

plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(sarsa_returns)

# with open('data/td_sarsa_agents.npy', 'rb') as g:
#         trained_agents = np.load(g, allow_pickle=True)

# ZERO_GREEDY = 0
# FIFTH_GREEDY = 0.2

# greedy_returns = []

# for agent in trained_agents:
#         agent.epsilon = FIFTH_GREEDY

#         greedy_policy = agent.get_current_policy()
#         states, actions, rewards = utl.generate_episode(greedy_policy, env)
#         greedy_returns.append(sum(rewards))

# print(f"Sarsa agents accumulated returns obtained in each episode run using {FIFTH_GREEDY}-greedy policy:", greedy_returns)


qlearning_returns = []
qlearning_agents = []

# for i in range(5):
#         info['seed'] = i
#         np.random.seed(info['seed'])
#         random.seed(info['seed'])
#         all_returns, agent = td_control(env, agent_class=QLearningAgent, info=info, num_episodes=num_episodes)
#         qlearning_returns.append(all_returns)
#         qlearning_agents.append(agent)

# with open('data/td_qlearning_returns.npy', 'wb') as f:
#         np.save(f, qlearning_returns)
# with open('data/td_qlearning_agents.npy', 'wb') as g:
#         np.save(g, qlearning_agents)

# plt.figure(figsize=(15, 7))
# plt.grid()
# utl.plot_many(qlearning_returns)

# with open('data/td_qlearning_agents.npy', 'rb') as g:
#         trained_agents = np.load(g, allow_pickle=True)

# ZERO_GREEDY = 0
# FIFTH_GREEDY = 0.2

# greedy_returns = []

# for agent in trained_agents:
#         agent.epsilon = FIFTH_GREEDY

#         greedy_policy = agent.get_current_policy()
#         states, actions, rewards = utl.generate_episode(greedy_policy, env)
#         greedy_returns.append(sum(rewards))

# print(f"Q-Learning agents accumulated returns obtained in each episode run using {FIFTH_GREEDY}-greedy policy:", greedy_returns)
