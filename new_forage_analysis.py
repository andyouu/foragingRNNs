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
import scipy
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


def data_creation(data_dir,load_folder, num_steps_exp, verbose, probs_task):
    """
    Perform general analysis on trained networks in the specified folder.
    Data is stored in CSV files for each subject and for aggregated results.
    
    Args:
        load_folder (str): Path to folder containing trained networks
        num_steps_exp (int): Number of steps to run the experiment
        verbose (bool): Whether to show detailed plots
        probs_task: Task probabilities configuration
        save_data (bool): Whether to save the collected data to CSV files
        load_data (bool): Whether to load previously saved data instead of running new simulations
        analysis_only (bool): Whether to only perform analysis on existing data (skip simulations)
    """
    print(load_folder)
    if not os.path.exists(load_folder):
        print(f"The directory {load_folder} does not exist.")
        return

    # Initialize storage for cross-subject analysis
    all_subjects_data = []
    subject_ids = []
    combined_data_file = os.path.join(data_dir, 'all_subjects_data.csv')
    for root, dirs, files in os.walk(load_folder):
        mice_counter = 0
        n_subjects = len(dirs)
        
        if n_subjects > 0:   
            for dir_name in dirs:
                prefix, seed = dir_name.rsplit('_', 1)
                subject_ids.append(seed)                    
                # Create environment
                task_env = 'ForagingBlocks-v0'
                env_kwargs, env = ft.create_env(
                    env_seed=env_seed, 
                    mean_ITI=mean_ITI, 
                    max_ITI=max_ITI,
                    fix_dur=fix_dur, 
                    dec_dur=dec_dur,
                    blk_dur=blk_dur, 
                    probs=probs_task, 
                    task=task_env,
                    variable_blk_dur=True
                )
                
                dir_path = os.path.join(root, dir_name)
                print(f"Found folder: {dir_path}")
                
                # Load network
                save_folder_net = dir_path
                net_pth_path = os.path.join(save_folder_net, 'net.pth')
                
                if os.path.exists(net_pth_path):
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
                    data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp, env=env, net=net)

                if verbose and seed == "570976":
                    ft.plot_task(env_kwargs=env_kwargs, data=data, num_steps=2000, save_folder=None)
                    plt.show()
                
                df = ft.dict2df(data)
                df['network_seed'] = seed
                # Calculate performance metrics
                all_subjects_data.append(df)
    combined_df = pd.concat(all_subjects_data, ignore_index=True,axis=0)
    combined_df.to_csv(combined_data_file, index=False)


def weights_computation(model, data_dir, glm_dir, n_regressors):
    df = pd.read_csv(data_dir, sep=',', low_memory=False)
    combined_glm_file = os.path.join(glm_dir, 'all_subjects_weights.csv')
    combined_glm_data = os.path.join(glm_dir, 'all_subjects_glm_regressors.csv')
    all_glms = []
    all_datas_regressors = []
    subjects = np.unique(df['network_seed']) 
    n_subjects = len(subjects)
    mice_counter = 0
    for net in subjects:
        try:                                        
            if model == 'glm_prob_r':
                GLM_df, regressors_string,df_regressors= glm_prob_r_analysis(df[df['network_seed'] == net], net,n_regressors)
                df_glm = GLM_df.copy()
                df_glm['seed'] = net
                df_regressors['seed'] = net
                df_glm['regressors_string'] = regressors_string
                #Set the indexes as a new column to facilitate the analysis
                df_reset = df_glm.reset_index()
                df_reset = df_reset.rename(columns={'index': 'regressor'})
                all_glms.append(df_reset)
                all_datas_regressors.append(df_regressors)
                
            
            elif model == 'glm_prob_switch':
                GLM_df, regressors_string, df_regressors = glm_switch_analysis(df[df['network_seed'] == net],net,n_regressors)
                df_glm = GLM_df.copy()
                df_glm['seed'] = net
                df_regressors['seed'] = net
                df_glm['regressors_string'] = regressors_string
                #Set the indexes as a new column to facilitate the analysis
                df_reset = df_glm.reset_index()
                df_reset = df_reset.rename(columns={'index': 'regressor'})
                all_glms.append(df_reset)
                all_datas_regressors.append(df_regressors)

            elif model == 'inference_based':
                inference_plot(df[df['network_seed'] == net])
                                                
            mice_counter += 1
            
        except np.linalg.LinAlgError as e:
            print(f"Singular matrix encountered for {net}: {str(e)}")
            continue  
        combined_glms = pd.concat(all_glms, ignore_index=True,axis=0)
        combined_glms.to_csv(combined_glm_file, index=False)
        combined_glms_reg = pd.concat(all_datas_regressors, ignore_index=True,axis=0)
        combined_glms_reg.to_csv(combined_glm_data, index=False)

def plotting_w(model,glm_dir, data_dir, n_regressors):
    #read the data with the weights
    df = pd.read_csv(glm_dir, sep=',', low_memory=False)
    #read the data with the original data
    orig_data = pd.read_csv(data_dir, sep=',', low_memory=False)
    subjects = np.unique(df['seed'])
    # Create figure layouts for results
    n_cols = int(np.ceil(len(subjects) / 2))
    f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=False)
    f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
    regressors_string = df.loc[0,'regressors_string']
    
    for mice_counter, net in enumerate(subjects):
        ax = axes[mice_counter//n_cols, mice_counter%n_cols]
        ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]
        ax.set_title(f'GLM weights: {net}')
        ax1.set_title(f'Psychometric Function: {net}')
    
        if model == 'glm_prob_r':
            plot_GLM_prob_r(ax, df[df['seed']== net], 1)
            psychometric_data(ax1,orig_data[orig_data['seed']== net],df[df['seed']== net], regressors_string,'choice')
            ax1.set_ylabel('Prob of going right')
        elif model == 'glm_prob_switch':
            plot_GLM_prob_switch(ax,  df[df['seed']== net], 1)
            psychometric_data(ax1, orig_data[orig_data['seed']== net], df[df['seed']== net],regressors_string,'switch_num')
            ax1.set_ylabel('Prob of switching')
            
        ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Evidence')
        ax1.legend(loc='upper left')
    plt.show()
    
    if model == 'glm_prob_r':
        plt.figure(figsize=(12, 6))
        regressor_list = [x.strip() for x in regressors_string.split(' + ')]
        avg_r_plus = []
        avg_r_minus = []
        for i,reg in enumerate(regressor_list):
            if 'r_plus' in reg: 
                avg_r_plus.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
            if 'r_minus' in reg: 
                avg_r_minus.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x_vals = np.arange(1, n_regressors)  # x-axis values

        ax.plot(x_vals, avg_r_plus, marker='o', color='indianred')
        ax.plot(x_vals, avg_r_minus, marker='o', color='teal')

        # Custom legend
        legend_handles = [
            mpatches.Patch(color='indianred', label='r+'),
            mpatches.Patch(color='teal', label='r-'),
        ]

        ax.legend(handles=legend_handles)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_ylabel('GLM weight')
        ax.set_xlabel('Previous trials')
        ax.set_title('Average GLM Coefficients Across Networks')
        plt.show()
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='regressor', y='coefficient', data=df, color='lightblue', width=0.6)
            # Overlay the individual data points
        sns.stripplot(x='regressor', y='coefficient', data=df, color='black', alpha=0.6, jitter=True)
        plt.show()


    elif model == 'glm_prob_switch':
        plt.figure(figsize=(12, 6))
        regressor_list = [x.strip() for x in regressors_string.split(' + ')]
        avg_rss_plus = []
        avg_rds_plus = []
        avg_rss_minus = []
        avg_last_trial = []
        for i,reg in enumerate(regressor_list):
            if 'rss_plus' in reg: 
                avg_rss_plus.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
            if 'rds_plus' in reg: 
                avg_rds_plus.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
            if 'rss_minus' in reg: 
                avg_rss_minus.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
            if 'last_trial' in reg: 
                avg_last_trial.append(df[df['regressor'].str.contains(reg, regex=False)]['coefficient'].mean())
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x_vals = np.arange(1, n_regressors)  # x-axis values

        ax.plot(x_vals, avg_rss_plus, marker='o', color='indianred')
        ax.plot(x_vals, avg_rds_plus, marker='o', color='orange')
        ax.plot(x_vals, avg_rss_minus, marker='o', color='teal')
        ax.axhline(avg_last_trial, color='black', linestyle='-', label='last_trial')

        # Custom legend
        legend_handles = [
            mpatches.Patch(color='indianred', label='rss+'),
            mpatches.Patch(color='teal', label='rss-'),
            mpatches.Patch(color='orange', label='rds+'),
            mpatches.Patch(color='black', label='last_trial')
        ]

        ax.legend(handles=legend_handles)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_ylabel('GLM weight')
        ax.set_xlabel('Previous trials')
        ax.set_title('Average GLM Coefficients Across Networks')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='regressor', y='coefficient', data=df, color='lightblue', width=0.6)
            # Overlay the individual data points
        sns.stripplot(x='regressor', y='coefficient', data=df, color='black', alpha=0.6, jitter=True)
        plt.show()
                                    

                
                    
def plotting_perf(data_dir):
    data = pd.read_csv(data_dir, sep=',', low_memory=False)
     # Remove invalid values
    block_values = np.unique(data['prob_r'])  # Get unique block identifiers
    
    # Store performance by block for this subject
    all_perf_by_block = []
    subjects = np.unique(data['network_seed'])
    for subj in subjects:
        subject_perf = {}
        data_s = data[data['network_seed']== subj]
        perf = np.array(data_s['perf'])
        perf = perf[perf != -1] 
        for blk in block_values:
            mask = (data_s['prob_r'] == blk)[:len(perf)]
            perf_cond = perf[mask]
            mean_perf_cond = np.mean(perf_cond) if len(perf_cond) > 0 else np.nan 
            subject_perf[blk] = mean_perf_cond
            
            print(f'block: {blk}')
            print(f'mean performance: {mean_perf_cond}')
        all_perf_by_block.append(subject_perf)
    plt.figure(figsize=(10, 6))
    # Convert performance data to dataframe
    perf_df = pd.DataFrame(all_perf_by_block)
    perf_df['subject'] = subjects
    perf_melted = pd.melt(perf_df, id_vars=['subject'], 
                            var_name='block_prob', value_name='performance')
    
    # Plot block performance
    sns.boxplot(x='block_prob', y='performance', data=perf_melted, color='lightgreen')
    sns.stripplot(x='block_prob', y='performance', data=perf_melted, color='black', alpha=0.5)
    
    # Add chance level line
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    
    plt.title('Performance Across Probability Blocks')
    plt.xlabel('Block Probability (Right)')
    plt.ylabel('Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
# define parameters configuration
    env_seed = 123
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
    [0.2, 0.8],[0.3, 0.7],[0.1, 0.9],[0.4, 0.6],[0.8, 0.2], [0.7, 0.3],[0.9, 0.1],[0.6, 0.4]])
    #seeds 42 and 13 and 100
    seed = 42
    np.random.seed(seed)
    probs_task = []
    for i in range(100):
        if(i%2 == 0):
            j = np.random.randint(0, 3)
            probs_task.append(blocks[j])
        else: 
            j = np.random.randint(4, 8)
            probs_task.append(blocks[j])

    print("Selected blocks:", probs_task)
    probs_net = np.array([[0.4, 0.6],[0.6, 0.4]])
    # to avaluate on the same enviroment than the training
    #probs_task = [np.array([0.3, 0.7]), np.array([0.7, 0.3])]
    #env.reset()
    #Change ForagingBlocks for whatever TASK teh network is doing
    folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_prb{probs_net[0][0]}{probs_net[0][1]}")
    redo = True
    # Check if analysis_results.pkl exists in the main folder
    model = 'glm_prob_r'
    data_dir = os.path.join(folder, f'analysis_data_{model}')

    #Control
    Redo_data = 0
    Redo_glm = 0
    Plot_weights = 1
    Plot_performance = 0

    if Redo_data or not os.path.exists(data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        else:
        # Clear existing data files if redoing
            for file in os.listdir(data_dir):
                if file.endswith('data.csv'):
                    os.remove(os.path.join(data_dir, file))
        data_creation(data_dir = data_dir, load_folder=folder, num_steps_exp=100000, verbose=False, probs_task=probs_task)
    combined_data_file = os.path.join(data_dir, 'all_subjects_data.csv')
    n_regressors = 10
    glm_dir = os.path.join(folder, f'{model}_weights_{n_regressors}')
    if Redo_glm or not os.path.exists(glm_dir):
        if not os.path.exists(glm_dir):
            os.makedirs(glm_dir)
        else:
        # Clear existing data files if redoing
            for file in os.listdir(data_dir):
                if file.endswith('weights.csv'):
                    os.remove(os.path.join(data_dir, file))
        weights_computation(model = model, data_dir = combined_data_file, glm_dir = glm_dir, n_regressors = n_regressors)

    combined_glm_file = os.path.join(glm_dir, 'all_subjects_weights.csv')
    combined_glm_data = os.path.join(glm_dir, 'all_subjects_glm_regressors.csv')
    if Plot_weights:
        plotting_w(model = model, glm_dir = combined_glm_file, data_dir=combined_glm_data, n_regressors = n_regressors)

    if Plot_performance:
        plotting_perf(data_dir = combined_data_file)
    #general_analysis(model = model,load_folder=folder, num_steps_exp=100000, verbose=False, probs_task=probs_task)
        # TODO: move inside general_analysis
        #save_general_analysis_results(sv_folder=folder, seeds=seeds, mean_perf_list=mean_perf_list,
        #                            mean_perf_smooth_list=mean_perf_smooth_list, iti_bins=iti_bins, 
        #                            mean_perf_iti=mean_perf_iti, GLM_coeffs=GLM_coeffs, net_nums=net_nums)
    