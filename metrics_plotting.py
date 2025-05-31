import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import patches as mpatches
import matplotlib.lines as mlines



def filter_df_performance(df,combined_data_file):
    underperforming_nets = []
    if os.path.exists(combined_data_file):
        data = pd.read_csv(combined_data_file)
        subjects = np.unique(data['network_seed'])
        block_values = np.unique(data['prob_r'])
        for subj in subjects:
            subject_perf = {}
            data_s = data[data['network_seed']== subj]
            perf = np.array(data_s['perf'])
            perf = perf[perf != -1] 
            mean_perf = np.sum(data_s['perf'])/len(data_s)
            print(f'AVERAGE performance: {mean_perf}')
            if(mean_perf < 0.55):
                print(subj)
                underperforming_nets.append(subj)
    else:
        print(f"Data file not found: {combined_data_file}")
    return underperforming_nets

def plot_perf_psychos(combined_data):
    data_total = combined_data
    
    # Set up figure for A0 poster with constrained layout
    plt.figure(figsize=(20, 18), constrained_layout=False)
    
    # Set font sizes
    title_fontsize = 28
    label_fontsize = 50
    legend_fontsize = 30
    tick_fontsize = 30
    linewidth = 5
    markersize = 14
    
    # Get all training blocks and sort them
    training_blocks = np.sort(np.unique(data_total['training_block']))
    
    # Initialize alpha value
    base_alpha = 1.0
    alpha_step = 0.15
    
    # Create colormap for different training blocks
    colors = plt.cm.viridis(np.linspace(0, 1, len(training_blocks)))
    
    for i, train_block in enumerate(training_blocks):
        data = data_total[data_total['training_block'] == train_block]
        block_values = np.unique(data['prob_r'])
        
        all_perf_by_block = {blk: [] for blk in block_values}
        
        subjects = np.unique(data['network_seed'])
        for subj in subjects:
            data_s = data[data['network_seed'] == subj]
            perf = np.array(data_s['perf'])
            perf = perf[perf != -1]
            for blk in block_values:
                mask = (data_s['prob_r'] == blk)[:len(perf)]
                perf_cond = perf[mask]
                mean_perf_cond = np.mean(perf_cond) if len(perf_cond) > 0 else np.nan
                transformed_perf = mean_perf_cond if blk > 0.5 else 1 - mean_perf_cond
                all_perf_by_block[blk].append(transformed_perf)
        
        block_probs = sorted(all_perf_by_block.keys())
        mean_perfs = [np.mean(all_perf_by_block[blk]) for blk in block_probs]
        
        current_alpha = max(0.3, base_alpha - (i * alpha_step))
        
        if train_block == 0:
            train_block_label = f'{train_block}/0.9'
        else: 
            train_block_label = f'{train_block}/{1-train_block}'
        
        plt.plot(block_probs, mean_perfs, 'o-', 
                color=colors[i], 
                alpha=current_alpha,
                linewidth=linewidth, 
                markersize=markersize,
                label=f'Training: {train_block_label}',
                markeredgecolor='black',
                markeredgewidth=1.5)
    
    plt.plot(block_probs, block_probs, 'k--', alpha=0.5, linewidth=linewidth)
    # Add chance level line
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, 
               linewidth=3)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, 
            linewidth=3)
    # Customize plot appearance with padding
    # plt.title('Average Probability Right Across Probability Blocks', 
    #          fontsize=title_fontsize, pad=25)  # Increased pad
    plt.xlabel('Block Prob of reward (Right)', 
              fontsize=label_fontsize, labelpad=20)  # Increased labelpad
    plt.ylabel('Average Prob of right', 
              fontsize=label_fontsize, labelpad=20)
    
    # Customize legend
    legend = plt.legend(loc='upper left', 
                      fontsize=legend_fontsize,
                      framealpha=0.9,
                      edgecolor='black',
                      facecolor='white',
                      bbox_to_anchor=(0.02, 0.98),
                      borderaxespad=0.5)
    
    # Make legend frame visible
    legend.get_frame().set_linewidth(2)
    
    # Customize ticks and grid
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.yticks(fontsize=tick_fontsize)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(pad=4.0)  # Increased padding
    plt.show()

def plot_performance_comparison(df_data,combined_data_file):    
    # Set up figure for A0 poster
    plt.figure(figsize=(24, 16))
    
    # Set font sizes
    title_fontsize = 55
    label_fontsize = 55
    tick_fontsize = 30
    legend_fontsize = 30
    
    # Get all training blocks and sort them
    training_blocks = np.sort(np.unique(df_data['training_block']))
    
    # Create colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(df_data['training_block']))))
    
    # Prepare data structures
    good_perfs = []
    bad_perfs = []
    boxplot_labels = []
    
    for i, train_block in enumerate(training_blocks):
        good_block_perfs = []
        bad_block_perfs = []           
        data = df_data[df_data['training_block'] == train_block]
        subjects = np.unique(data['network_seed'])
        for subj in subjects:
            subject_perf = {}
            data_s = data[data['network_seed']== subj]
            perf = np.array(data_s['perf'])
            perf = perf[perf != -1] 
            mean_perf = np.sum(data_s['perf'])/len(data_s)
            print(f'AVERAGE performance: {mean_perf}')
            if(mean_perf < 0.55):
                bad_block_perfs.append(mean_perf)
            else:
                good_block_perfs.append(mean_perf)
        # Append to lists   
        good_perfs.append(good_block_perfs)
        bad_perfs.append(bad_block_perfs)
        
        # Create labels
        if train_block == 0:
            train_block_label = f'{train_block}/0.9'
        else: 
            train_block_label = f'{train_block}/{1-train_block:.1f}'
        boxplot_labels.append(train_block_label)
    
    # Create boxplot for good networks
    boxprops = [{'facecolor': c, 'alpha': 1-(0.15*i)} for i,c in enumerate(colors)]

    bp = plt.boxplot(good_perfs, positions=np.arange(len(training_blocks))+1,
                    patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='none'))  # Initialize

    # Apply styles
    for patch, props in zip(bp['boxes'], boxprops):
        patch.update(props)
    
    # Style other elements
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=2)
    
    # Add scattered points for bad networks
    for i in range(len(training_blocks)):
        x_pos = np.random.normal(i+1, 0.05, size=len(bad_perfs[i]))
        plt.scatter(x_pos, bad_perfs[i], color='red', alpha=0.7, 
                   s=100, edgecolor='black', linewidth=1.5,
                   label='Poor Performers' if i == 0 else "")
    
    # Add chance level line
    plt.axhline(0.55, color='gray', linestyle='--', linewidth=3, alpha=0.7)
    
    #increase y-axis ticks size in the plot
    plt.yticks(fontsize=tick_fontsize)

    # Customize plot
    plt.xticks(np.arange(len(training_blocks))+1, boxplot_labels, fontsize=tick_fontsize)
    plt.xlabel('Pre-training condition', fontsize=label_fontsize, labelpad=20)
    plt.ylabel('Average Performance', fontsize=label_fontsize, labelpad=20)
    #plt.title('Network Performance Comparison', fontsize=title_fontsize, pad=20)
    plt.ylim(0.4, 0.85)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], alpha=0.6, label='Good Performers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=15, markeredgecolor='black', label='Poor Performers'),
        plt.Line2D([0], [0], color='gray', linestyle='--', 
                  label='Discrimination Threshold', linewidth=3),
    ]
    
    plt.legend(handles=legend_elements, fontsize=legend_fontsize,
              loc='upper right', framealpha=0.9)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(blocks, metrics_data, model_names):
    """
    Create separate plots for BIC and Accuracy with legends inside the plots
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
    """
    # Create a consistent color palette
    palette = sns.color_palette("husl", len(model_names))
    
    # Convert blocks to readable labels
    block_labels = [f"{p[0]}/{p[1]}" for p in blocks]
    
    # Set font sizes
    title_fontsize = 50
    label_fontsize = 50
    legend_fontsize = 20  # Slightly smaller for inside placement
    tick_fontsize = 25
    # Plot n_trials
    plt.figure(figsize=(10, 6))  # Slightly smaller for single plot
    plot_metric(blocks, metrics_data, model_names, 'n_trials', 'n_trials', 
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    
    # Plot BIC
    plt.figure(figsize=(10, 6))  # Slightly smaller for single plot
    plot_metric(blocks, metrics_data, model_names, 'BIC', 'BIC', 
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plot_metric(blocks, metrics_data, model_names, 'accuracy', 'Accuracy', 
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()
    
    #Plot Log Likelihood
    plt.figure(figsize=(10, 6))
    plot_metric(blocks, metrics_data, model_names, 'log_likelihood_per_obs', 'Log Likelihood per Obs',
               palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize)
    plt.tight_layout()
    plt.show()

def plot_metric(blocks, metrics_data, model_names, metric, ylabel, 
                palette, block_labels, title_fontsize, label_fontsize, legend_fontsize, tick_fontsize):
    """Helper function to plot a single metric with internal legend"""
    # Prepare data
    plot_data = []
    for model_idx, model in enumerate(model_names):
        for block_idx, block in enumerate(blocks):
            values = metrics_data[model][metric][block_idx]
            for val in values:
                plot_data.append({
                    'Model': model,
                    'Probability': block_labels[block_idx],
                    'Value': val,
                    'Color': palette[model_idx]
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create plot
    ax = plt.gca()
    
    # Create boxplot
    sns.boxplot(
        x='Probability', 
        y='Value', 
        hue='Model',
        data=df_plot,
        palette=palette,
        width=0.6,
        linewidth=1.5
    )
    
    # Add individual data points
    sns.stripplot(
        x='Probability',
        y='Value',
        hue='Model',
        data=df_plot,
        dodge=True,
        palette=palette,
        alpha=0.5,
        edgecolor='gray',
        linewidth=0.5,
        jitter=True,
        size=4  # Slightly smaller points
    )
    
    # Format plot
    #ax.set_title(ylabel, fontsize=title_fontsize, pad=10)
    ax.set_xlabel('Pre-training condition', fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Customize legend - placed inside plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        labels, 
        title='Model',
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc='best',  # Changed to upper right inside
        framealpha=1,
        edgecolor='black'
    )
    
    # Add grid and clean borders
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)

def plot_metrics_variance(blocks, metrics_data, model_names):
    """
    Compute the performance increase for each metric as prob[0] increases across blocks
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
        
    Returns:
        Dictionary containing performance increase information for each model and metric
    """
    # Sort blocks by prob[0] to ensure increasing order
    sorted_blocks = sorted(blocks, key=lambda x: x[1])
    block_labels = [f"{p[0]}/{p[1]}" for p in sorted_blocks]
    
    # Removed AIC from metrics
    metrics = ['log_likelihood_per_obs', 'BIC', 'accuracy']
    metric_titles = {
        'log_likelihood_per_obs': 'Log Likelihood per Observation',
        'BIC': 'Bayesian Information Criterion',
        'accuracy': 'Accuracy'
    }
    
    results = {
        model: {
            metric: {
                'values': None,
                'normalized_values': None
            }
            for metric in metrics
        }
        for model in model_names
    }
    
    # Set up figure for A0 poster - 1 row of 3 square plots
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))  # 36 inches wide for A0
    axes = axes.flatten()
    
    # Set font sizes for poster
    title_fontsize = 24
    label_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 16
    linewidth = 4
    markersize = 12
    
    palette = sns.color_palette("husl", len(model_names))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, model in enumerate(model_names):
            # Get values in order of increasing prob[0]
            values = [
                np.mean(metrics_data[model][metric][block_idx])
                for block_idx, block in enumerate(sorted_blocks)
            ]
            # change order in values
            values = [values[-block_idx] for block_idx in range(1,len(blocks)+1)]
            
            # Normalize by subtracting the first value
            values_norm = (values - values[0])/abs(values[0])
            
            # Store results
            results[model][metric]['values'] = values
            results[model][metric]['normalized_values'] = values_norm
            
            # Plot the normalized values with larger elements
            ax.plot(block_labels, values_norm, 
                   marker='o', 
                   color=palette[j],
                   label=model,
                   linewidth=linewidth,
                   markersize=markersize)
        
        ax.set_title(metric_titles[metric], fontsize=title_fontsize)
        ax.set_xlabel('Probability Blocks', fontsize=label_fontsize)
        ax.set_ylabel('Normalized Evolution', fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Make plot square
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    return results

def plot_combined_switch_analysis(data_dir, window, probs):
    df = data_dir
    subjects = np.unique(df['network_seed'])
    df['choice'] = df['actions']-2
    #keep only the right-left actions
    df = df[df['choice'] >= 0]
    switchy = 1
    if switchy == 1:
        df['choice_1'] = df['choice'].shift(1)
        df.loc[(df['choice'] == df['choice_1']), 'switch_num'] = 0
        df.loc[(df['choice'] != df['choice_1']), 'switch_num'] = 1
    else:
        df['prob_r_1'] = df['prob_r'].shift(1)
        df.loc[(df['prob_r'] == df['prob_r_1']), 'switch_num'] = 0
        df.loc[(df['prob_r'] != df['prob_r_1']), 'switch_num'] = 1
    # Calculate fraction of correct responses
    df['fraction_of_correct_responses'] = np.where(
            ((df['prob_r'] >= 0.5) & (df['choice'] == 1)) |
            ((df['prob_r'] < 0.5) & (df['choice'] == 0)),
            1, 0
        )
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(24, 16))
    title_fontsize = 50
    label_fontsize = 50
    legend_fontsize = 30  # Slightly smaller for inside placement
    tick_fontsize = 30
    
    # Define colors and alphas for different training blocks
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(df['training_block']))))
    alphas = [0.3, 0.6]  # Two different alpha values for each block
    
    for i, train in enumerate(np.unique(df['training_block'])):
        # Initialize lists to store all aligned data
        t_df = df[df['training_block'] == train].reset_index(drop=True)
        if switchy == 1:
            t_df = df[:int(0.6*len(t_df))]  # Keep only the first 80% of trials
        all_outcome = []
        all_switches = []
        subject_means = []
        
        for subj in subjects:
            subj_df = t_df[t_df['network_seed'] == subj].reset_index(drop=True)
            
            # Calculate subject mean
            subject_mean = subj_df['fraction_of_correct_responses'].mean()
            subject_means.append(subject_mean)
            
            # ----- Switch indices -----
            switch_idx = subj_df.index[subj_df['switch_num'] == 1]
            
            # ----- Switch-triggered outcome -----
            for idx in switch_idx:
                if idx - window < 0 or idx + window >= len(subj_df):
                    continue

                lags = np.arange(-window, window + 1)
                outcome_vals = subj_df.iloc[idx - window:idx + window + 1]['fraction_of_correct_responses'].values
                switch_vals = subj_df.iloc[idx - window:idx + window + 1]['switch_num'].values

                for lag, o, s in zip(lags, outcome_vals, switch_vals):
                    all_outcome.append({'lag': lag, 'fraction_of_correct_responses': o})
                    all_switches.append({'lag': lag, 'switch': s})
        
        # Convert to DataFrames
        outcome_df = pd.DataFrame(all_outcome)
        switch_df = pd.DataFrame(all_switches)
        
        # Calculate overall means
        overall_subject_mean = np.mean(subject_means)
        
        # --- Plot outcome curve ---
        outcome_mean = outcome_df.groupby('lag')['fraction_of_correct_responses'].mean()
        outcome_sem = outcome_df.groupby('lag')['fraction_of_correct_responses'].sem()
        if train == 0:
            label = f'{train}/0.9'
        else:
            label = f'{train}/{1-train}'
        # First line for this training block (lower alpha)
        ax.errorbar(outcome_mean.index, outcome_mean.values, yerr=outcome_sem.values,
                    fmt='o-', capsize=3, color=colors[i],
                    label=f'Training:({label})',linewidth=5)
        
        # Second line for this training block (higher alpha)
        # ax.errorbar(outcome_mean.index, outcome_mean.values, yerr=None,
        #             fmt='-', color=colors[i],
        #             label=f'Average Outcome ({label})')
        
        # --- Plot switch probability ---
        switch_prob = switch_df.groupby('lag')['switch'].mean()
        switch_sem = switch_df.groupby('lag')['switch'].sem()
        
        # First line for this training block (lower alpha)
        if switchy == 1:
            switch_label = f'P(switch) ({label})'
            # ax.errorbar(switch_prob.index, switch_prob.values, yerr=switch_sem.values,
            # fmt='o-', capsize=3, alpha=alphas[0], color=colors[i],
            # label=switch_label)
        else:
            switch_label = f'P(change_block) ({label})'
        switch_prob[0] = 0.5*switch_prob[-1] + 0.5*switch_prob[1]
    if switchy == 1:
        ax.set_xlabel('Trials from switch',size=label_fontsize)
        ax.axvline(0, linestyle='--', color='black', label='Switch')
    else:
        ax.set_xlabel('Trials from block transition',size=label_fontsize)
        ax.axvline(0, linestyle='--', color='black', label='Block change')

        
        # Second line for this training block (higher alpha)
        # ax.errorbar(switch_prob.index, switch_prob.values, yerr=None,
        #             fmt='-', alpha=alphas[1], color=colors[i],
        #             label=f'P(switch) ({label})')
    
    # --- Reference lines ---
    ax.axhline(0.5, linestyle='--', color='gray', label='Chance')
    ax.set_ylabel('Average Performance',size=label_fontsize)
    ax.set_ylim(0.3, 0.7)
    plt.yticks(fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    ax.legend(loc='lower left',fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 38
    blocks = np.array([
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]#,[2,2]
    ])
    
    # Store metrics for each model
    metrics_data = {
        'glm_prob_switch': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'n_trials': [],
            'accuracy': []
        },
        'glm_prob_r': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'n_trials': [],
            'accuracy': []
        },
        'inference_based': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'n_trials': [],
            'accuracy': []
        },
        # 'inference_based_v2': {
        #     'log_likelihood_per_obs': [],
        #     'BIC': [],
        #     'AIC': [],
        #     'accuracy': []
        # }
    }
    data_file_total = []
    for model in ['glm_prob_switch', 'glm_prob_r', 'inference_based']:#, 'inference_based_v2']:

        for probs_net in blocks:
            #selecto optimal number of trials back for each model and pre-training condition
            if model == 'glm_prob_r':
                n_regressors = 7
                if probs_net[0] == 0.4:
                    n_regressors = 10
            if model == 'glm_prob_switch':
                n_regressors = 10
                if probs_net[0] == 0.0:
                    n_regressors = 7
            if model == 'inference_based':
                n_back = 3
            folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                     f"d{dec_dur}_prb{probs_net[0]}{probs_net[1]}")
            #easy way to include custom-trained nets
            if probs_net[0] == 2:
                seed_task = 13
                folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_"f"prb_task_seed_{seed_task}")
            if(model == 'inference_based'):
                glm_dir = os.path.join(folder, f'{model}_weights_{n_back}')
            elif(model == 'inference_based_v2'):
                glm_dir = os.path.join(folder, f'{model}_weights')
            else:
                glm_dir = os.path.join(folder, f'{model}_weights_{n_regressors}')
            data_dir = os.path.join(folder, f'analysis_data_{model}')
            combined_glm_metrics = os.path.join(glm_dir, 'all_subjects_glm_metrics.csv')
            combined_data_file = os.path.join(data_dir, 'all_subjects_data.csv')

            if os.path.exists(combined_glm_metrics):
                df = pd.read_csv(combined_glm_metrics)
                bad_nets = filter_df_performance(df,combined_data_file)
                df = df[~df['seed'].isin(bad_nets)]
                #To avooid repetition of data
                if model == 'inference_based':
                    df_data = pd.read_csv(combined_data_file)
                    df_data['training_block'] = probs_net[0]
                    df_data = df_data[~df_data['network_seed'].isin(bad_nets)]
                    data_file_total.append(df_data)
                #To just plot one point for each seed, comment to see all cross-validation cases
                df = df.groupby('seed').mean()
                # Store metrics
                metrics_data[model]['log_likelihood_per_obs'].append(df['log_likelihood_per_obs'].values)
                metrics_data[model]['n_trials'].append(df['log_likelihood'].values/df['log_likelihood_per_obs'].values)
                metrics_data[model]['BIC'].append(df['BIC'].values)
                metrics_data[model]['AIC'].append(df['AIC'].values)
                metrics_data[model]['accuracy'].append(df['accuracy'].values)
            else:
                print(f"Metrics file not found: {combined_glm_metrics}")
                # Append empty arrays if data is missing
                metrics_data[model]['log_likelihood_per_obs'].append(np.array([]))
                metrics_data[model]['BIC'].append(np.array([]))
                metrics_data[model]['AIC'].append(np.array([]))
                metrics_data[model]['accuracy'].append(np.array([]))
    
    # Generate plots
    # Combine data for plotting
    combined_data = pd.concat(data_file_total, ignore_index=True)
    #plot_combined_switch_analysis(combined_data, 10, blocks[0][0])
    #plot_perf_psychos(combined_data)
    plot_metrics_comparison(blocks, metrics_data, ['glm_prob_switch', 'glm_prob_r', 'inference_based'])#, 'inference_based_v2'])
    plot_performance_comparison(combined_data,combined_data_file)
    plot_perf_psychos(combined_data)
    plot_metrics_variance(blocks, metrics_data, ['glm_prob_switch', 'glm_prob_r', 'inference_based'])#, 'inference_based_v2'])





