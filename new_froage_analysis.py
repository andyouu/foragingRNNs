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



def general_analysis(load_folder, env,num_steps_exp, verbose):
    #iterate over the folders inside the network's folder
    print(load_folder)
    if not os.path.exists(load_folder):
        print(f"The directory {load_folder} does not exist.")
    else:
        for root, dirs, files in os.walk(load_folder):
            mice_counter = 0
            n_subjects = len(dirs)
            if n_subjects > 0:
                n_cols = int(np.ceil(n_subjects / 2))
                f, axes = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
                f1, axes1 = plt.subplots(2, n_cols, figsize=(5*n_cols-1, 8), sharey=True)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    print(f"Found folder: {dir_path}")
                    #construc the path and load the data
                    save_folder_net = dir_path
                    net_pth_path = os.path.join(save_folder_net, 'net.pth')
                    #get the seed from the file name
                    prefix, seed = dir_name.rsplit('_', 1)
                    if os.path.exists(net_pth_path):
                        # If net.pth exists, load it directly
                        net = Net(input_size=NET_KWARGS['input_size'],
                                hidden_size=NET_KWARGS['hidden_size'],
                                output_size=env.action_space.n, seed=seed)

                        # Move network to the device (CPU or GPU)
                        net = torch.load(save_folder_net+'/net.pth', map_location=DEVICE, weights_only=False)
                        net = net.to(DEVICE)  # Move it explicitly
                    else:
                        print('No net with name ', net_pth_path, ' exists')

                    with torch.no_grad():
                        data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp, env=env,
                                                    net=net)
                    
                    df = ft.dict2df(data)
                    df_glm, regressors_string = GLM_regressors(df)
                    #df_80, df_20 = select_train_sessions(df_glm_mice) # cross-reference may not be neded here bc train and test data is different
                    mM_logit = smf.logit(formula='choice ~ ' + regressors_string, data=df_glm).fit()
                    GLM_df = pd.DataFrame({
                        'coefficient': mM_logit.params,
                        'std_err': mM_logit.bse,
                        'z_value': mM_logit.tvalues,
                        'p_value': mM_logit.pvalues,
                        'conf_Interval_Low': mM_logit.conf_int()[0],
                        'conf_Interval_High': mM_logit.conf_int()[1]
                        })
                    # subplot title with name of mouse
                    ax = axes[mice_counter//n_cols, mice_counter%n_cols]
                    ax1 = axes1[mice_counter//n_cols, mice_counter%n_cols]

                    ax.set_title(f'GLM weights: {seed}')
                    ax1.set_title(f'Psychometric Function: {seed}')
                    plot_GLM(ax, GLM_df,1)
                    ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax1.axvline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax1.set_xlabel('Evidence')
                    ax1.set_ylabel('Prob of switching')
                    ax1.legend(loc='upper left')
                    mice_counter += 1
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
    blk_dur = 25
    probs = np.array([[0.2, 0.8],[0.8, 0.2]])
    ENV_KWARGS = {'dt': TRAINING_KWARGS['dt'], 'timing':
                    {'ITI': ngym.ngym_random.TruncExp(mean_ITI, 100, max_ITI),
                        # mean, min, max
                        'fixation': fix_dur, 'decision': dec_dur},
                    # Decision period}
                    'rewards': {'abort': 0., 'fixation': 0., 'correct': 1.}}
    
    #perquè definim això? No fa falta!!
    TRAINING_KWARGS['classes_weights'] =\
        torch.tensor([w_factor*TRAINING_KWARGS['dt']/(mean_ITI), #why are this dependent on the iTI and the fixed duration and not somewhat univ constant
                    w_factor*TRAINING_KWARGS['dt']/fix_dur, 2, 2])      #it is balaced because it lasts longer? Is ir somehow normalized?
    # call function to sample
    env_kwargs, env = ft.create_env(env_seed=env_seed, mean_ITI=mean_ITI, max_ITI=max_ITI,
                                        fix_dur=fix_dur, dec_dur=dec_dur,
                                        blk_dur=blk_dur, probs=probs, task = TASK)
    # set seed
    #env.seed(env_seed)
    env.get_wrapper_attr('seed')
    #env.reset()
    NET_KWARGS = {'hidden_size': 128,
                    'action_size': env.action_space.n,
                    'input_size': env.observation_space.n}
    TRAINING_KWARGS['env_kwargs'] = ENV_KWARGS
    TRAINING_KWARGS['net_kwargs'] = NET_KWARGS
    # create folder to save data based on parameters
    #Change ForagingBlocks for whatever TASK teh network is doing
    folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_prb{probs[0][0]}{probs[0][1]}")
    redo = True
    # Check if analysis_results.pkl exists in the main folder
    if not os.path.exists(f'{folder}/analysis_results.pkl') or redo:
        general_analysis(load_folder=folder,env = env, num_steps_exp=100000, verbose=True)
        # TODO: move inside general_analysis
        #save_general_analysis_results(sv_folder=folder, seeds=seeds, mean_perf_list=mean_perf_list,
        #                            mean_perf_smooth_list=mean_perf_smooth_list, iti_bins=iti_bins, 
        #                            mean_perf_iti=mean_perf_iti, GLM_coeffs=GLM_coeffs, net_nums=net_nums)
    