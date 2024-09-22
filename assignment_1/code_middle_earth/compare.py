import numpy as np
from heroes import Heroes
from eps_greedy import eps_greedy
from ucb import ucb
from boltzmann import boltzmann
from gradient_bandit import gradient_bandit
from helpers import run_trials, save_results_plots


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    results_list = []

    # Best eps-greedy 
    eps_greedy_best_eps = 0.6          # Modify this param
    eps_greedy_best_init_val = 0.      # Modify this param
    rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                heroes=heroes, bandit_method=eps_greedy, 
                                                                eps=eps_greedy_best_eps, init_value=eps_greedy_best_init_val)
    results_list.append({
        "exp_name": f"eps_greedy-eps={eps_greedy_best_eps}-init_val={eps_greedy_best_init_val}",
        "reward_rec": rew_rec,
        "average_rew_rec": avg_ret_rec,
        "tot_reg_rec": tot_reg_rec,
        "opt_action_rec": opt_act_rec
    })


    # Best UCB
    ucb_best_c = 10                    # Modify this param
    ucb_best_init_value = 0.0          # Modify this param
    rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                heroes=heroes, bandit_method=ucb, 
                                                                c=ucb_best_c, init_value=ucb_best_init_value)
    results_list.append({
        "exp_name": f"ucb-c={ucb_best_c}-init_val={ucb_best_init_value}",
        "reward_rec": rew_rec,
        "average_rew_rec": avg_ret_rec,
        "tot_reg_rec": tot_reg_rec,
        "opt_action_rec": opt_act_rec
    })

    # Best Boltzmann
    boltzmann_best_tau = 0.5          # Modify this param
    boltzmann_best_init_val = 0.0     # Modify this param
    rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                heroes=heroes, bandit_method=boltzmann, 
                                                                tau=boltzmann_best_tau, init_value=boltzmann_best_init_val)
    results_list.append({
        "exp_name": f"boltzmann-tau={boltzmann_best_tau}-init_val={boltzmann_best_init_val}",
        "reward_rec": rew_rec,
        "average_rew_rec": avg_ret_rec,
        "tot_reg_rec": tot_reg_rec,
        "opt_action_rec": opt_act_rec
    })

    # Best Gradient Bandit
    gradient_bandit_best_alpha = 2.75                            # Modify this param
    gradient_bandit_use_baseline = False # True or False         # Modify this param
    rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                heroes=heroes, bandit_method=gradient_bandit, 
                                                                alpha=gradient_bandit_best_alpha, use_baseline=gradient_bandit_use_baseline)
    results_list.append({
        "exp_name": f"gradient_bandit-alpha={gradient_bandit_best_alpha}-use_baseline={gradient_bandit_use_baseline}",
        "reward_rec": rew_rec,
        "average_rew_rec": avg_ret_rec,
        "tot_reg_rec": tot_reg_rec,
        "opt_action_rec": opt_act_rec
    })

    # Save results
    save_results_plots(results_list, plot_title="Ultimate Showdown: Tuning Parameters and Comparing Methods",
                       results_folder='results', pdf_name='final_comparison.pdf')

