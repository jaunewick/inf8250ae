import matplotlib.pyplot as plt
import os
import numpy as np


def run_trials(number_of_trials, heroes, bandit_method, **kwargs):
    """
    Runs a specified bandit method for a given number of trials and returns the averaged results.

    Parameters:
    - number_of_trials (int): The number of times to run the bandit method.
    - heroes (Heroes): An instance of the Heroes class, representing the available heroes for the bandit problem.
    - bandit_method (function): The bandit method to be used (e.g., eps_greedy). 
    - kwargs: Additional arguments required by the bandit method.

    Returns:
    - rew_rec (numpy.ndarray): The average rewards recorded over all trials.
    - avg_ret_rec (numpy.ndarray): The average of average returns recorded over all trials.
    - tot_reg_rec (numpy.ndarray): The average total regret recorded over all trials.
    - opt_act_rec (numpy.ndarray): The average percentage of optimal actions taken over all trials.
    """

    rew_rec = np.zeros(heroes.total_quests)
    avg_ret_rec = np.zeros(heroes.total_quests)
    tot_reg_rec = np.zeros(heroes.total_quests)
    opt_act_rec = np.zeros(heroes.total_quests)

    for _ in range(number_of_trials):
        # Initialize the heroes for a new simulation
        heroes.init_heroes()

        # Run the bandit method with the given kwargs
        cur_rew_rec, cur_avg_ret_rec, cur_tot_reg_rec, cur_opt_act_rec = bandit_method(heroes=heroes, **kwargs)

        rew_rec += np.array(cur_rew_rec)
        avg_ret_rec += np.array(cur_avg_ret_rec)
        tot_reg_rec += np.array(cur_tot_reg_rec)
        opt_act_rec += np.array(cur_opt_act_rec)

    rew_rec /= number_of_trials
    avg_ret_rec /= number_of_trials
    tot_reg_rec /= number_of_trials
    opt_act_rec /= number_of_trials

    return rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec


def save_results_plots(experiments, plot_title='Experiment Results', results_folder='results', pdf_name='experiment_results.pdf'):
    """
    Create a 2x2 plot of results from multiple experiments and save it as a PDF.

    :param experiments: List of experiments where each experiment is a dictionary
                        containing 'exp_name', 'reward_rec', 'average_rew_rec',
                        'tot_reg_rec', and 'opt_action_rec'.
    :param plot_title: The title for the plot.
    :param results_folder: Directory where the PDF will be saved. It will be created if it does not exist.
    :param pdf_name: Name of the final PDF.
    """
    
    # Ensure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set up the figure and axes for the 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(plot_title, fontsize=16)

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))

    # Plot each experiment
    for exp, color in zip(experiments, colors):
        exp_name = exp['exp_name']
        reward_rec = exp['reward_rec']
        average_rew_rec = exp['average_rew_rec']
        tot_reg_rec = exp['tot_reg_rec']
        opt_action_rec = exp['opt_action_rec']

        # Plot reward
        axs[0, 0].plot(reward_rec, label=exp_name, color=color)
        axs[0, 0].set_title('Reward Over Time')
        axs[0, 0].set_xlabel('Number of Attempts')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()

        # Plot average reward
        axs[0, 1].plot(average_rew_rec, label=exp_name, color=color)
        axs[0, 1].set_title('Average Reward Over Time')
        axs[0, 1].set_xlabel('Number of Attempts')
        axs[0, 1].set_ylabel('Average Reward')
        axs[0, 1].legend()

        # Plot total regret
        axs[1, 0].plot(tot_reg_rec, label=exp_name, color=color)
        axs[1, 0].set_title('Total Regret Over Time')
        axs[1, 0].set_xlabel('Number of Attempts')
        axs[1, 0].set_ylabel('Total Regret')
        axs[1, 0].legend()

        # Plot percentage of optimal actions
        axs[1, 1].plot(opt_action_rec, label=exp_name, color=color)
        axs[1, 1].set_title('Percentage of Optimal Actions Over Time')
        axs[1, 1].set_xlabel('Number of Attempts')
        axs[1, 1].set_ylabel('Percentage of Optimal Actions')
        axs[1, 1].legend()

    # Save the plot as a PDF
    pdf_path = os.path.join(results_folder, pdf_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(pdf_path, format='pdf')
    plt.close()

    print(f"Results saved to {pdf_path}")
