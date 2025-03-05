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

def load_net(save_folder,performance,take_best):
    # check if net.pth exists in the folder (for nets that have not been saved
    # several times during training)
    net_pth_path = os.path.join(save_folder, 'net.pth')
    if os.path.exists(net_pth_path):
        # If net.pth exists, load it directly
        net = torch.load(net_pth_path, weights_only= False)
        network_number = 0
    else:
        # If net.pth doesn't exist, find the newest net,
        # which is the file with the highest number
        net_files = [f for f in os.listdir(save_folder) if 'net' in f]
        # find the number of the newest net file, being the file names net0,
        # net1, net2, etc.
        net_files = np.array([int(f.split('net')[1].split('.pth')[0]) for f in
                              net_files])
        if take_best:
            # find the best net based on performance
            best_net = np.argmax(performance)
            # find closest network in net_files
            index = np.argmin(np.abs(net_files - best_net))
            network_number = net_files[index]
        else:
            net_files.sort()
            network_number = net_files[-1]
        net_file = 'net'+str(network_number)+'.pth'
        net_path = os.path.join(save_folder, net_file)
        net = torch.load(net_path)
    return net, network_number


def general_analysis(load_folder, file, env, take_best, num_steps_exp,
                        verbose):
    net_nums = []
    mean_perf_list = []
    mean_perf_smooth_list = []
    #iterate over the folders inside the network's folder
    for root, dirs, files in os.walk(load_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"Found folder: {dir_path}")
            #construc the path and load the data
            save_folder_net = dir_path
            data_training = np.load(save_folder_net + '/training_data.npz',
                                    allow_pickle=True)
            # get mean performance from data and smoothen it
            mean_performance = data_training['mean_perf_list']
            roll = 20
            mean_performance_smooth = np.convolve(mean_performance,
                                                np.ones(roll)/roll, mode='valid')
            mean_perf_smooth_list.append(mean_performance_smooth)
            #load the network with mean_performance_smooth as performance
            net, network_number = load_net(save_folder=save_folder_net,
                                        performance=mean_performance_smooth, #why are we using the mean_pernformance_smooth?
                                        take_best=take_best)
            net_nums.append(network_number)
            #Test the net selected
            data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp,env=env, net=net)
            perf = np.array(data['perf'])
            perf = perf[perf != -1]
            mean_perf = np.mean(perf)
            mean_perf_list.append(mean_perf)
            
            #perform the analysis for those networks with higher performance than a set threshold
            if mean_perf > PERF_THRESHOLD:
                df = ft.dict2df(data)
                f, ax = plt.subplots(1, 1, figsize=(10, 6))
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
                plot_GLM(ax, GLM_df,1)

if __name__ == '__main__':
# define parameters configuration
    PERF_THRESHOLD = 0.7
    env_seed = 123
    total_num_timesteps = 6000
    num_periods = 2000
    env_seed = 123
    num_periods = 40
    TRAINING_KWARGS['num_periods'] = num_periods
    # create folder to save data based on env seed
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2'
    # main_folder = '/home/molano/Dropbox/Molabo/foragingRNNs/' # '/home/molano/foragingRNNs_data/nets/'
   # main_folder = '/home/manuel/foragingRNNs/files/'
    # Set up the task
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 25
    probs = np.array([0.2, 0.8])
    ENV_KWARGS = {'dt': TRAINING_KWARGS['dt'], 'timing':
                    {'ITI': ngym.ngym_random.TruncExp(mean_ITI, 100, max_ITI),
                        # mean, min, max
                        'fixation': fix_dur, 'decision': dec_dur},
                    # Decision period}
                    'rewards': {'abort': 0., 'fixation': 0., 'correct': 1.}}
    
    #perquè definim això?
    TRAINING_KWARGS['classes_weights'] =\
        torch.tensor([w_factor*TRAINING_KWARGS['dt']/(mean_ITI), #why are this dependent on the iTI and the fixed duration and not somewhat univ constant
                    w_factor*TRAINING_KWARGS['dt']/fix_dur, 2, 2])      #it is balaced because it lasts longer? Is ir somehow normalized?
    # call function to sample
    env = gym.make(TASK, **ENV_KWARGS)
    env = pass_reward.PassReward(env)
    env = pass_action.PassAction(env)
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
                    f"d{dec_dur}_prb{probs[0]}{probs[1]}")
    filename = folder+'/training_data_w1e-02.csv'
    redo = True
    # Check if analysis_results.pkl exists in the main folder
    if not os.path.exists(f'{folder}/analysis_results.pkl') or redo:
        general_analysis(load_folder=folder, file=filename, env=env, take_best=True, num_steps_exp=100000,
                        verbose=True)
        # TODO: move inside general_analysis
        #save_general_analysis_results(sv_folder=folder, seeds=seeds, mean_perf_list=mean_perf_list,
        #                            mean_perf_smooth_list=mean_perf_smooth_list, iti_bins=iti_bins, 
        #                            mean_perf_iti=mean_perf_iti, GLM_coeffs=GLM_coeffs, net_nums=net_nums)
    