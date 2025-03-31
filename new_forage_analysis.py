import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd
import numpy as np
import os
import glob
import tkinter as tk
import gymnasium as gym
import seaborn as sns
from scipy.optimize import curve_fit
import neurogym as ngym
import statsmodels.formula.api as smf


from neurogym.wrappers import pass_reward, pass_action, side_bias
import forage_training as ft
from GLM_related_fucntions import *
from inference_based_functions import inference_plot

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# name of the task on the neurogym library
#TASK = 'Foraging-v0'
TASK = 'ForagingBlocks-v0'
# TASK = 'PerceptualDecisionMaking-v0' 
TRAINING_KWARGS = {'dt': 100,
                   'lr': 1e-2,
                   'seq_len': 300,
                   'TASK': TASK}


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        # build a recurrent neural network with a single
        # recurrent layer and rectified linear units
        # set seed for weights
        torch.manual_seed(seed)
        self.vanilla = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # If hidden state is not provided, initialize it
        if hidden is None:
            hidden = torch.zeros(1, TRAINING_KWARGS['seq_len'],
                                 self.hidden_size)
        # get the output of the network for a given input
        out, _ = self.vanilla(x, hidden)
        x = self.linear(out)
        return x, out



def general_analysis(model,load_folder, num_steps_exp, verbose, probs_task):
    """
    Perform general analysis on trained networks in the specified folder.
    
    Args:
        load_folder (str): Path to folder containing trained networks
        env: Environment configuration
        num_steps_exp (int): Number of steps to run the experiment
        verbose (bool): Whether to show detailed plots
        probs_task: Task probabilities configuration
    """
    print(load_folder)
    if not os.path.exists(load_folder):
        print(f"The directory {load_folder} does not exist.")
        return  # Added return to exit if folder doesn't exist
    
    # Iterate over all subdirectories in the load_folder
    for root, dirs, files in os.walk(load_folder):
        mice_counter = 0
        n_subjects = len(dirs)
        
        if n_subjects > 0:
            # Create figure layouts for results
            n_cols = int(np.ceil(n_subjects / 2))
            f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=False)
            f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
            
            for dir_name in dirs:
                try:
                    prefix, seed = dir_name.rsplit('_', 1)
                    
                    # Create environment with specified parameters
                    #Set the task for the enviroment (need not be the same than for the net)
                    task_env = 'ForagingBlocks-v0'
                    env_kwargs, env = ft.create_env(
                        env_seed=env_seed, 
                        mean_ITI=mean_ITI, 
                        max_ITI=max_ITI,
                        fix_dur=fix_dur, 
                        dec_dur=dec_dur,
                        blk_dur=blk_dur, 
                        probs=probs_task, 
                        task= task_env,
                        #set to true to consider duration-variable blocks
                        variable_blk_dur = False
                    )
                    
                    dir_path = os.path.join(root, dir_name)
                    print(f"Found folder: {dir_path}")
                    
                    # Load network
                    save_folder_net = dir_path
                    net_pth_path = os.path.join(save_folder_net, 'net.pth')
                    
                    if os.path.exists(net_pth_path):
                        # Initialize and load network
                        NET_KWARGS = {'hidden_size': 128,
                                        'action_size': env.action_space.n,
                                        'input_size': env.observation_space.n}
                        net = Net(
                            input_size=NET_KWARGS['input_size'],
                            hidden_size=NET_KWARGS['hidden_size'],
                            output_size=env.action_space.n, 
                            seed=seed
                        )
                        net = torch.load(net_pth_path, map_location=DEVICE, weights_only=False)
                        net = net.to(DEVICE)
                    else:
                        print(f'No net with name {net_pth_path} exists')
                        continue
                    
                    # Run agent in environment
                    with torch.no_grad():
                        data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp, env=env,net=net)

                    if verbose and seed == "570976":
                        ft.plot_task(env_kwargs=env_kwargs,data=data,num_steps=2000,save_folder=None)
                        plt.show()
                                    
                    # Calculate performance metrics
                    perf = np.array(data['perf'])
                    perf = perf[perf != -1]  # Remove invalid values
                    block_values = np.unique(data['prob_r'])  # Get unique block identifiers
                    for blk in block_values:
                        mask = (data['prob_r'] == blk)[:len(perf)]
                        perf_cond = perf[mask]
                        mean_perf_cond = np.mean(perf_cond) if len(perf_cond) > 0 else np.nan
                        
                        print(f'block: {blk}')
                        print(f'mean performance: {mean_perf_cond}')
                    mean_perf = data['mean_perf']
                    # Set up subplots
                    ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                    ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]
                                    
                    if model == 'glm_prob_r':
                        GLM_df = glm_prob_r_analysis(data,seed,mean_perf)
                        
                        ax.set_title(f'GLM weights: {seed}, perf: {mean_perf:.2f}')
                        ax1.set_title(f'Psychometric Function: {seed}')
                    
                        # Plot results
                        plot_GLM_prob_r(ax, GLM_df, 1)
                        ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.set_xlabel('Evidence')
                        ax1.set_ylabel('Prob of going right')
                        ax1.legend(loc='upper left')
                    
                    elif model == 'glm_prob_switch':
                        GLM_df = glm_switch_analysis(data,seed,mean_perf)
                        
                        ax.set_title(f'GLM weights: {seed}, perf: {mean_perf:.2f}')
                        ax1.set_title(f'Psychometric Function: {seed}')
                    
                        # Plot results
                        plot_GLM_prob_switch(ax, GLM_df, 1)
                        ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.set_xlabel('Evidence')
                        ax1.set_ylabel('Prob of switching')
                        ax1.legend(loc='upper left')

                    elif model == 'inference_based':
                        df = ft.dict2df(data)
                        inference_plot(ax,df)
                        ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                        ax1.set_xlabel('Evidence')
                        ax1.set_ylabel('Prob of going right')
                        ax1.legend(loc='upper left')
                            

                            
                    mice_counter += 1
                    
                except Exception as e:
                    print(f"Error processing {dir_name}: {e}")
                    continue    
            plt.tight_layout()
            plt.show()
                    

if __name__ == '__main__':
# define parameters configuration
    PERF_THRESHOLD = 0.7
    env_seed = 123
    total_num_timesteps = 6000
    num_periods = 2000
    num_periods = 40
    TRAINING_KWARGS['num_periods'] = num_periods
    # create folder to save data based on env seed
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    # main_folder = '/home/molano/Dropbox/Molabo/foragingRNNs/' # '/home/molano/foragingRNNs_data/nets/'
   # main_folder = '/home/manuel/foragingRNNs/files/'
    # Set up the task
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 38
    probs_task = []
    blocks = np.array([
    [0.2, 0.8], [0.8, 0.2],
    [0.3, 0.7], [0.7, 0.3],
    [0.1, 0.9], [0.9, 0.1],
    [0.4, 0.6], [0.6, 0.4]
    ])
    #seeds 42 and 13 and 100
    seed = 13
    np.random.seed(seed)
    probs_task = []
    for i in range(100):
        j = np.random.randint(0, 8)
        probs_task.append(blocks[j])

    print("Selected blocks:", probs_task)
    probs_net = np.array([[0, 0.9],[0.9, 0]])
    # to avaluate on the same enviroment than the training
    probs_task = [np.array([0.3, 0.7]), np.array([0.7, 0.3])]
    #env.reset()
    #Change ForagingBlocks for whatever TASK teh network is doing
    folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_prb{probs_net[0][0]}{probs_net[0][1]}")
    redo = True
    # Check if analysis_results.pkl exists in the main folder
    if not os.path.exists(f'{folder}/analysis_results.pkl') or redo:
        general_analysis(model = 'glm_prob_r',load_folder=folder, num_steps_exp=100000, verbose=False, probs_task=probs_task)
        # TODO: move inside general_analysis
        #save_general_analysis_results(sv_folder=folder, seeds=seeds, mean_perf_list=mean_perf_list,
        #                            mean_perf_smooth_list=mean_perf_smooth_list, iti_bins=iti_bins, 
        #                            mean_perf_iti=mean_perf_iti, GLM_coeffs=GLM_coeffs, net_nums=net_nums)
    