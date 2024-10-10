from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from environment import RaceTrack


State = tuple[int, int, int, int]
Action = int
StateAction = tuple[int, int, int, int, Action]
ValueDict = dict[State, float]
ActionValueDict = dict[StateAction, float]
Policy = Callable[[State], int]
DistributionPolicy = Callable[[State], list[float]]



def plot_many(experiments: list[list[float]], label: Optional[str] = None, color: Optional[str] = None) -> None:
    """
        Plots results from a series of experiments

        :param list[list[float]] experiments: each element of the list is one experiment run
        :param str label: label of the overall plot
        :param str color: color string for the graph
    """
    mean_exp = np.mean(experiments, axis=0)
    std_exp = np.std(experiments, axis=0)
    plt.plot(mean_exp, color=color, label=label)
    plt.fill_between(range(len(experiments[0])), mean_exp + std_exp, mean_exp - std_exp, color=color, alpha=0.1)
    plt.show()


def init_q_and_v() -> tuple[list[StateAction], ActionValueDict, list[State], ValueDict]:
    """
    Utility functions that returns the list of all states and state-actions tuples, as well as empty value dictionnaries
    and action-value dictionnaries (th ekeys are either sates or state_actions and the values are all zero)

    :return state_action_array: a list of all state-action pairs
    :return state_action_returns: a dictionary with keys = (s,a) and values = Q(s,a) initialized at 0
    :return state_array: a list of all states
    :return state_returns: a dictionary with keys = s and values = v(s) initialized at []
    """
    state_action_array = list(product(np.arange(12), np.arange(10),
                                      np.arange(-4, 1, 1), np.arange(-4, 5, 1), np.arange(0, 9, 1)))
    state_array = list(product(np.arange(12), np.arange(10),
                               np.arange(-4, 1, 1), np.arange(-4, 5, 1)))
    state_action_returns = {}
    state_returns = {}
    for sa in state_action_array:
        state_action_returns[sa] = 0.
    for s in state_array:
        state_returns[s] = 0.
    return state_action_array, state_action_returns, state_array, state_returns


def qs_from_q(state_action_values: ActionValueDict = None) -> dict[State, list[float]]:
    """
    Change the structure {(s,a): Q(s,a)} to {s: [Q(s,a_1), Q(s,a_2),... Q(s,a_n)]}

    :param dict[StateAction, float] state_action_values: dictionary with keys = (s,a) and values = Q(s,a)
    :return state_values (dict): each key is a state, the corresponding value is a list of Q-value estimates,
    one for each action
    """
    all_states = init_q_and_v()[2]
    state_values = {state: np.zeros(9) for state in all_states}
    for sa in state_action_values:
        state_values[sa[:-1]][sa[-1]] = state_action_values[sa]
    return state_values


def random_argmax(value_list: list) -> np.ndarray:
    """ a random tie-breaking argmax """
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def make_eps_greedy_policy(state_action_values: ActionValueDict, epsilon: float) -> Policy:
    """
        Creates an epsilon-greedy policy from given q-values. This function returns a function that takes a state
        as input and ouputs the action chosen from the policy

        :param dict[StateAction, float] state_action_values: dictionary containing the q-values (values of the dict) for each state-action configuration (keys of the dict)
        :param flat epsilon: probability of taking a random action

        :return policy ((tuple -> int)): function taking a state as input and returning a sampled action to take
    """
    n_actions = 9
    state_values = qs_from_q(state_action_values)

    def policy(state: State) -> Action:
        action = 0

        # TO IMPLEMENT
        # ----------------------------------

        # ----------------------------------

        return action

    return policy


def generate_episode(policy: Policy, env: RaceTrack) -> tuple[list[State], list[Action], list[float]]:
    """
        Returns the visited states, taken actions and received rewards from an agent during an episode in the environment
        `env` following policy `policy`

        :param (tuple -> int) policy: policy taking a state as an input and outputs a given action
        :param RaceTrack env: the RaceTrack environment

        :return states (list[tuple]): the sequence of states in the generated episode
        :return actions (list[int]): the sequence of actions in the generated episode
        :return rewards (list[float]): the sequence of rewards in the generated episode
    """
    states = []
    rewards = []
    actions = []

    # TO IMPLEMENT
    # --------------------------------

    # --------------------------------

    return states, actions, rewards


def make_eps_greedy_policy_distribution(state_action_values: ActionValueDict, epsilon: float) -> DistributionPolicy:
    """
        Creates an epsilon-greedy policy from given q-values. This function returns a function that takes a state
        as input and ouputs the list of probabilities of taking any action from this state

        :param dict[StateAction, float] state_action_values:  dictionary containing the q-values (values of the dict) for each state-action configuration (keys of the dict)
        :param float epsilon: probability of taking a random action
        :return policy ((tuple -> list[float])): function taking a state as input and returning a sampled action to take
    """
    n_actions = 9
    state_values = qs_from_q(state_action_values)

    def policy(state: State) -> list[float]:
        action_probabilities = np.zeros(9)

        # TO IMPLEMENT
        # ----------------------------------

        # ----------------------------------

        return action_probabilities

    return policy


def convert_to_sampling_policy(distribution_policy: DistributionPolicy) -> Policy:
    """
        Converts a policy that returns the list of probabilities of taking each actions into a policy that samples actions
        from that probability distribution

        :param (tuple -> list[float]) distribution_policy:  policy to transform
        :return policy ((tuple -> int)): transformed policy
    """
    def policy(state):
        action_probabilities = distribution_policy(state)
        return np.random.choice(len(action_probabilities), p=action_probabilities)

    return policy
