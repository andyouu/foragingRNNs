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
from matplotlib.lines import Line2D


from neurogym.wrappers import pass_reward, pass_action, side_bias
import forage_training as ft
from GLM_related_fucntions import *
from inference_based_functions import *

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

def raster_plot(df):
    # Compute probabilities
    df['time'] = range(len(df))
    
    # Probability of action = 3 (right) with 5-trial rolling window
    
    # Reward probability
    prob_reward = df['prob_r']
    
    # Create figure with primary and secondary axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot probabilities on primary y-axis
    ax1.plot(df['time'], prob_reward, 'b-', label='Reward Probability', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(-0.1, 1.1)  # Give some padding for markers
    
    # Create secondary axis for action markers
    ax2 = ax1.twinx()
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Left (2)', 'Right (3)'])
    ax2.set_ylabel('Action Choice', rotation=270, labelpad=15)
    
    # Plot action choices
    for i, row in df.iterrows():
        if row['actions'] in [2, 3]:
            y_pos = 0 if row['actions'] == 2 else 1
            if row['actions'] == row['gt']:
                # Correct action - larger green circle
                ax2.plot(row['time'], y_pos, 'go', markersize=8, alpha=0.7)
            else:
                # Incorrect action - smaller red x
                ax2.plot(row['time'], y_pos, 'rx', markersize=6, alpha=0.7)
            if row['reward'] == 1:
                #Indicate the side where the reward was?
                ax2.plot(row['time'], y_pos, 'g.', markersize=6, alpha=0.7)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    
    plt.title('Action Selection Probability and Choices Over Time')
    plt.tight_layout()
    plt.show()

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
                    #set this parametert to True to always choose the action with highest probability and to False to sample the actions with probability
                    deterministic = False
                    data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp, env=env, net=net, deterministic = deterministic)

                if verbose:
                    df_raster = ft.dict2df(data)
                    raster_plot(df_raster[:300])
                    ft.plot_task(env_kwargs=env_kwargs, data=data, num_steps=100, save_folder=None)
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
    combined_glm_metrics = os.path.join(glm_dir, 'all_subjects_glm_metrics.csv')
    all_glms = []
    all_datas_regressors = []
    all_metrics = []
    subjects = np.unique(df['network_seed']) 
    n_subjects = len(subjects)
    mice_counter = 0
    for net in subjects:
        try:                                        
            if model == 'glm_prob_r':
                GLM_df, regressors_string,df_regressors,df_metrics= glm_prob_r_analysis(df[df['network_seed'] == net], net,n_regressors)
                df_glm = GLM_df.copy()
                df_glm['seed'] = net
                df_regressors['seed'] = net
                df_metrics['seed'] = net
                df_glm['regressors_string'] = regressors_string
                #Set the indexes as a new column to facilitate the analysis
                df_reset = df_glm.reset_index()
                df_reset = df_reset.rename(columns={'index': 'regressor'})
                all_glms.append(df_reset)
                all_datas_regressors.append(df_regressors)
                all_metrics.append(df_metrics)
                
            
            elif model == 'glm_prob_switch':
                GLM_df, regressors_string, df_regressors, df_metrics = glm_switch_analysis(df[df['network_seed'] == net],net,n_regressors)
                df_glm = GLM_df.copy()
                df_glm['seed'] = net
                df_regressors['seed'] = net
                df_metrics['seed'] = net
                df_glm['regressors_string'] = regressors_string
                #Set the indexes as a new column to facilitate the analysis
                df_reset = df_glm.reset_index()
                df_reset = df_reset.rename(columns={'index': 'regressor'})
                all_glms.append(df_reset)
                all_datas_regressors.append(df_regressors)
                all_metrics.append(df_metrics)

            elif model == 'inference_based':
                n_back = 5
                df_regressors, df_metrics = inference_data(df[df['network_seed'] == net],n_back=n_back)
                df_regressors['seed'] = net
                df_metrics['seed'] = net
                all_datas_regressors.append(df_regressors)
                all_metrics.append(df_metrics)                  
            mice_counter += 1
            
        except np.linalg.LinAlgError as e:
            print(f"Singular matrix encountered for {net}: {str(e)}")
            continue  
        if(model == 'glm_prob_r' or model == 'glm_prob_switch'):
            combined_glms = pd.concat(all_glms, ignore_index=True,axis=0)
            combined_glms.to_csv(combined_glm_file, index=False)
        combined_glms_reg = pd.concat(all_datas_regressors, ignore_index=True,axis=0)
        combined_glms_reg.to_csv(combined_glm_data, index=False)
        combined_metrics = pd.concat(all_metrics,ignore_index=True,axis=0)
        combined_metrics.to_csv(combined_glm_metrics, index=False)

def plotting_w(model,glm_dir, data_dir, n_regressors):
    #read the data with the weights
    if(model == 'glm_prob_r' or model == 'glm_prob_switch'):
        df = pd.read_csv(glm_dir, sep=',', low_memory=False)
    #read the data with the original data
    orig_data = pd.read_csv(data_dir, sep=',', low_memory=False)
    subjects = np.unique(orig_data['seed'])
    # Create figure layouts for results
    n_cols = int(np.ceil(len(subjects) / 2))
    f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=False)
    f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
    
    for mice_counter, net in enumerate(subjects):
        ax = axes[mice_counter//n_cols, mice_counter%n_cols]
        ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]
        ax.set_title(f'GLM weights: {net}')
        ax1.set_title(f'Psychometric Function: {net}')
    
        if model == 'glm_prob_r':
            regressors_string = df.loc[0,'regressors_string']
            plot_GLM_prob_r(ax, df[df['seed']== net], 1)
            psychometric_data(ax1,orig_data[orig_data['seed']== net],df[df['seed']== net], regressors_string,'choice')
            ax1.set_ylabel('Prob of going right')
        elif model == 'glm_prob_switch':
            regressors_string = df.loc[0,'regressors_string']
            plot_GLM_prob_switch(ax,  df[df['seed']== net], 1)
            psychometric_data(ax1, orig_data[orig_data['seed']== net], df[df['seed']== net],regressors_string,'switch_num')
            ax1.set_ylabel('Prob of switching')
        elif model == 'inference_based':
            inference_plot(ax1, orig_data[orig_data['seed']== net])
            ax1.set_ylabel('Prob of going right')
            
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
            if blk > 0.5:
                subject_perf[blk] = mean_perf_cond
            else:
                subject_perf[blk] = 1 - mean_perf_cond
            
            print(f'block: {blk}')
            print(f'mean performance: {mean_perf_cond}')
        all_perf_by_block.append(subject_perf)
    plt.figure(figsize=(10, 6))
    # Convert performance data to dataframe
    perf_df = pd.DataFrame(all_perf_by_block)
    perf_df['subject'] = subjects
    perf_melted = pd.melt(perf_df, id_vars=['subject'], 
                            var_name='block_prob', value_name='prob_r')
    
    # Plot block prob_r
    sns.boxplot(x='block_prob', y='prob_r', data=perf_melted, color='lightgreen')
    sns.stripplot(x='block_prob', y='prob_r', data=perf_melted, color='black', alpha=0.5)
    
    # Add chance level line
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    
    plt.title('Average probability right Across Probability Blocks')
    plt.xlabel('Block Probability of reward(Right)')
    plt.ylabel('Average probability of going right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_switching_evidence_summary_v4(data_dir,model):
    df = pd.read_csv(data_dir, sep=',', low_memory=False)
    subjects = np.unique(df['seed'])
    # Create figure layouts for results
    
    for mice_counter, seed in enumerate(subjects):
        df_s = df[df['seed']== seed]
        if model == 'glm_prob_switch':
            y_true = df_s['switch_num']
        elif model == 'glm_prob_r':
            y_true = df_s['choice']
        y_pred_prob = df_s['pred_prob'].dropna()

        #get number of trials needed to compute the first predicted value

        lag_trials = len(y_true) - len(y_pred_prob)
        df_s = df_s.reset_index(drop=True)
        filtered_df = df_s[lag_trials:].copy()
        
        #select windows of trials to avaluate
        sessions = [np.array([20,70]), np.array([7000,7050]), np.array([1031,1112])]
        for session in sessions:
            session_data = filtered_df[session[0]:session[1]].copy()
            session_data = session_data.reset_index(drop=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                            gridspec_kw={'height_ratios': [1.2, 1.2]})

            prob_r_values = session_data['prob_r'].values
            trials = np.arange(session[0],session[1])

            block_starts = [0]
            for i in range(1, len(prob_r_values)):
                if prob_r_values[i] != prob_r_values[i - 1]:
                    block_starts.append(i)
            block_starts.append(len(prob_r_values))

            for i in range(len(block_starts) - 1):
                start_idx = block_starts[i]
                end_idx = block_starts[i + 1]

                start_trial = trials[start_idx] - 0.5
                end_trial = trials[end_idx - 1] + 0.5
                prob = prob_r_values[start_idx]
                center_trial = (start_trial + end_trial) / 2

                #beware of the probabilitites thing, we are plotting them contrarily to what de datas show
                color = '#9c36b5' if prob > 0.5 else '#2b8a3e'
                ax1.axvspan(start_trial, end_trial, ymin=0.72, ymax=0.78, color=color, alpha=0.7)
                ax1.text(center_trial, 0.60, f'{prob:.2f}', color=color,
                         fontsize=9, ha='center', va='top', fontweight='bold')
            session_data['trial'] = trials
            for _, row in session_data.iterrows():
                trial = row['trial']
                choice = row['choice']
                outcome = row['outcome_bool']
                if model == 'glm_prob_switch':
                    switch_num = row['switch_num']
                    
                    color = '#40c057' if choice == 0 else '#ae3ec9'
                    ypos = 0.0 if choice == 0 else 0.3
                    length = 0.2 if outcome == 1 else 0.1
                    ax1.vlines(trial, ypos, ypos + length, color=color, linewidth=1.2, alpha=0.85)
                    if switch_num == 1:
                        ax1.vlines(trial, -0.2, -0.3, color='black', linewidth=1.4)
                elif model == 'glm_prob_r':                    
                    color = '#40c057' if choice == 0 else '#ae3ec9'
                    ypos = 0.0 if choice == 0 else 0.3
                    length = 0.2 if outcome == 1 else 0.1
                    ax1.vlines(trial, ypos, ypos + length, color=color, linewidth=1.2, alpha=0.85)
            ax1.set_ylim(-0.4, 1.1)
            ax1.set_yticks([])
            ax1.set_title(f'Network {seed}, Session {session}', fontsize=12)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)

            # === PROB SWITCH ===
            ax2.plot(session_data['trial'], session_data['pred_prob'],
                    color='black', linestyle='-', linewidth=1.8, alpha=0.9)

            if(model == 'glm_prob_r'):
                ax2.set_ylabel('Prob(Right)')
                ax2.axhline(0.5, linestyle='--', color='red', linewidth=1)
            elif model == 'glm_prob_switch':
                ax2.set_ylabel('Prob(Switch)')
            ax2.set_xlabel('Trial')
            ax2.set_ylim(0, 1)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            if model == 'glm_prob_switch':
                switch_trials = session_data[session_data['switch_num'] == 1]['trial']

                # Plot vertical lines on both subplots
                for trial in switch_trials:                
                    # Lower plot (probability)
                    ax2.axvline(trial, color='#ff6b6b', linestyle=':', 
                            linewidth=1.2, alpha=0.4)


        # === LEGENDA comune ===
            custom_lines = [
                Line2D([0], [0], color='black', lw=2, label='Prob(Switch)'),
                Line2D([0], [0], color='#9c36b5', lw=4, label='Reward Right'),
                Line2D([0], [0], color='#2b8a3e', lw=4, label='Reward Left'),
                Line2D([0], [0], color='#40c057', lw=3, label='Left choice'),
                Line2D([0], [0], color='#ae3ec9', lw=3, label='Right choice'),
                Line2D([0], [0], color='black', lw=2, label='Switch (tick)')
            ]
            fig.legend(handles=custom_lines, frameon=False, fontsize=9, loc='upper right')

            fig.suptitle(f'Sessions Summary trials {session[0],session[1]})',
                        fontsize=16, y=1.02)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.35)
            plt.show()

def plot_combined_switch_analysis(data_dir, window, probs):
    df = pd.read_csv(data_dir, sep=',', low_memory=False)
    subjects = np.unique(df['seed'])
    n = len(subjects)
    # Create figure layouts for results
    n_cols = int(np.ceil(len(subjects) / 2))
    df['fraction_of_correct_responses'] = np.where(
            ((df['prob_r'] >= 0.5) & (df['choice'] == 1)) |
            ((df['prob_r'] < 0.5) & (df['choice'] == 0)),
            1, 0
        )
    fig, axs = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=False)
    axs = axs.flatten()

    for i, subj in enumerate(subjects):
        ax = axs[i]
        subj_df = df[df['seed'] == subj].reset_index(drop=True)

        # ----- Switch indices -----
        switch_idx = subj_df.index[subj_df['switch_num'] == 1]
        subject_mean = subj_df['fraction_of_correct_responses'].mean()

        # ----- Switch-triggered outcome -----
        aligned_outcome = []
        aligned_switches = []

        for idx in switch_idx:
            if idx - window < 0 or idx + window >= len(subj_df):
                continue

            lags = np.arange(-window, window + 1)
            outcome_vals = subj_df.iloc[idx - window:idx + window + 1]['fraction_of_correct_responses'].values
            switch_vals = subj_df.iloc[idx - window:idx + window + 1]['switch_num'].values

            for lag, o, s in zip(lags, outcome_vals, switch_vals):
                aligned_outcome.append({'lag': lag, 'fraction_of_correct_responses': o})
                aligned_switches.append({'lag': lag, 'switch': s})

        # --- DF per outcome e switch ---
        outcome_df = pd.DataFrame(aligned_outcome)
        switch_df = pd.DataFrame(aligned_switches)

        # --- Plot outcome curve ---
        outcome_mean = outcome_df.groupby('lag')['fraction_of_correct_responses'].mean()
        outcome_sem = outcome_df.groupby('lag')['fraction_of_correct_responses'].sem()
        ax.errorbar(outcome_mean.index, outcome_mean.values, yerr=outcome_sem.values,
                    fmt='o-', capsize=3, label='Outcome (mean ± SEM)', color='blue')

        # --- Plot switch probability ---
        switch_prob = switch_df.groupby('lag')['switch'].mean()
        switch_sem = switch_df.groupby('lag')['switch'].sem()
        ax.errorbar(switch_prob.index, switch_prob.values, yerr=switch_sem.values,
                    fmt='o-', capsize=3, label='P(switch)', color='orange')

        # --- Linee di riferimento ---
        ax.axvline(0, linestyle='--', color='black', label='Switch')
        ax.axhline(0.5, linestyle='--', color='gray', label='Chance')
        ax.axhline(subject_mean, linestyle=':', color='green', label='Subject Mean')

        ax.set_title(f'Subject {subj}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability / Outcome')
        ax.set_xlabel('Trial lag')
        ax.legend()

    for j in range(n, len(axs)):
        fig.delaxes(axs[j])  # Rimuove subplot vuoti

    plt.suptitle(f'Switch-Triggered Analysis per Subject, probs [{probs},{ 1-probs}]', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    seed = 13
    np.random.seed(seed)
    probs_task = []
    for i in range(100):
        if(i%2 == 0):
            j = np.random.randint(0, 4)
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
    model = 'inference_based'
    data_dir = os.path.join(folder, f'analysis_data_{model}')

    #Control
    Redo_data = 1
    Redo_glm = 1
    Plot_weights = 1
    Plot_performance = 0
    Plot_raster = 0
    Trig_switch = 1
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
    combined_glm_metrics = os.path.join(glm_dir, 'all_subjects_glm_metrics.csv')
    if Plot_weights:
        plotting_w(model = model, glm_dir = combined_glm_file, data_dir=combined_glm_data, n_regressors = n_regressors)

    if Plot_performance:
        plotting_perf(data_dir = combined_data_file)
    if Plot_raster:
        plot_switching_evidence_summary_v4(data_dir = combined_glm_data, model = model)

    if Trig_switch and model == 'glm_prob_switch':
        plot_combined_switch_analysis(data_dir = combined_glm_data, window=10, probs= probs_net[0][0])