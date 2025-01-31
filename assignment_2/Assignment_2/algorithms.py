import numpy as np

import utils as utl
from environment import RaceTrack
from utils import Action, ActionValueDict, DistributionPolicy, State, StateAction


def fv_mc_estimation(states : list[State], actions: list[Action], rewards: list[float], discount: float) -> ActionValueDict:
    """
        Runs Monte-Carlo prediction for given transitions with the first visit heuristic for estimating the values

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
    """
    visited_sa_returns = {}

    # TO IMPLEMENT
    # --------------------------------
    G = 0
    first_visit = {}

    for t in reversed(range(len(states))):
        sa = (*states[t], actions[t])

        G = discount * G + rewards[t]

        if sa not in first_visit:
            first_visit[sa] = t
            visited_sa_returns[sa] = G
    # --------------------------------

    return visited_sa_returns


def fv_mc_control(env: RaceTrack, epsilon: float, num_episodes: int, discount: float) -> tuple[ActionValueDict, list[float]]:
    """
        Runs Monte-Carlo control, using first-visit Monte-Carlo for policy evaluation and regular policy improvement

        :param RaceTrack env: environment on which to train the agent
        :param float epsilon: epsilon value to use for the epsilon-greedy policy
        :param int num_episodes: number of iterations of policy evaluation + policy improvement
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
        :return all_returns (list[float]): list of all the cumulative rewards the agent earned in each episode
    """
    # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    # TO IMPLEMENT
    # --------------------------------
    for _ in range(num_episodes):
        policy = utl.make_eps_greedy_policy(state_action_values, epsilon)
        states, actions, rewards = utl.generate_episode(policy, env)

        visited_sa_returns = fv_mc_estimation(states, actions, rewards, discount)
        for sa in visited_sa_returns:
            if sa not in all_state_action_values:
                all_state_action_values[sa] = []
            all_state_action_values[sa].append(visited_sa_returns[sa])

        for sa in all_state_action_values:
            state_action_values[sa] = np.mean(all_state_action_values[sa])

        all_returns.append(np.sum(rewards))
    # --------------------------------

    return state_action_values, all_returns


def is_mc_estimate_with_ratios(
    states: list[State],
    actions: list[Action],
    rewards: list[float],
    target_policy: DistributionPolicy,
    behaviour_policy: DistributionPolicy,
    discount: float
) -> dict[StateAction, list[tuple[float, float]]]:
    """
        Computes Monte-Carlo estimated q-values for each state in an episode in addition to the importance sampling ratio
        associated to that state

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param (int -> list[float]) target_policy: The initial target policy that takes in a state and returns
                                            an action probability distribution (the one we are  learning)
        :param (int -> list[float]) behavior_policy: The behavior policy that takes in a state and returns
                                            an action probability distribution
        :param float discount: discount factor

        :return state_action_returns_and_ratios (dict[tuple,list[tuple]]):
        Keys are all the states visited in the input episode
        Values is a list of tuples. The first index of the tuple is
        the IS estimate of the discounted returns
        of that state in the episode. The second index is the IS ratio
        associated with each of the IS estimates.
        i.e: if state (2, 0, -1, 1) is visited 3 times in the episode and action '7' is always taken in that state,
        state_action_returns_and_ratios[(2, 0, -1, 1, 7)] should be a list of 3 tuples.
    """
    state_action_returns_and_ratios = {}

    # TO IMPLEMENT
    # --------------------------------
    G = 0
    W = 1

    for t in reversed(range(len(states))):
        sa = (*states[t], actions[t])

        G = discount * G + rewards[t]

        if sa not in state_action_returns_and_ratios:
            state_action_returns_and_ratios[sa] = []
        state_action_returns_and_ratios[sa].append((G, W))

        W *= target_policy(states[t])[actions[t]] / behaviour_policy(states[t])[actions[t]]
    # --------------------------------

    return state_action_returns_and_ratios


def ev_mc_off_policy_control(env: RaceTrack, behaviour_policy: DistributionPolicy, epsilon: float, num_episodes: int, discount: float):
    # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    # TO IMPLEMENT
    # --------------------------------
    for _ in range(num_episodes):
        target_policy = utl.make_eps_greedy_policy_distribution(state_action_values, epsilon)
        states, actions, rewards = utl.generate_episode(utl.convert_to_sampling_policy(behaviour_policy), env)

        state_action_returns_and_ratios = is_mc_estimate_with_ratios(
            states,
            actions,
            rewards,
            target_policy,
            behaviour_policy,
            discount
        )
        for sa in state_action_returns_and_ratios:
            if sa not in all_state_action_values:
                all_state_action_values[sa] = []
            for G, W in state_action_returns_and_ratios[sa]:
                all_state_action_values[sa].append(G * W)

        for sa in all_state_action_values:
            state_action_values[sa] = np.mean(all_state_action_values[sa])

        all_returns.append(np.sum(rewards))
    # --------------------------------

    return state_action_values, all_returns
