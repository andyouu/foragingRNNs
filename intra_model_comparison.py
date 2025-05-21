import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from new_forage_analysis import weights_computation

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
            #print(f'AVERAGE performance: {mean_perf}')
            if(mean_perf < 0.55):
                #print(subj)
                underperforming_nets.append(subj)
    else:
        print(f"Data file not found: {combined_data_file}")
    return underperforming_nets

def plot_metrics_comparison(blocks, metrics_data, n_regressors):
    """
    Create separate plots for each metric comparing models and probability conditions
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
    """
    # Create a consistent color palette
    palette = sns.color_palette("husl", len(n_regressors))
    
    # Convert blocks to readable labels
    block_labels = [f"{p[0]}/{p[1]}" for p in blocks]
    
    # Plot each metric separately
    metrics = ['log_likelihood_per_obs', 'BIC', 'accuracy']
    y_labels = ['Log Likelihood per Obs', 'BIC', 'Accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(40, 36))
    axes = axes.flatten()
    
    for i, (metric, ylabel) in enumerate(zip(metrics, y_labels)):
        ax = axes[i]
        
        # Prepare data for this metric
        plot_data = []
        for model_idx, n_reg in enumerate(n_regressors):
            for block_idx, block in enumerate(blocks):
                values = metrics_data[n_reg][metric][block_idx]
                for val in values:
                    plot_data.append({
                        'N_reg': n_reg if model == 'inference_based' else n_reg-1,
                        'Probability': block_labels[block_idx],
                        'Value': val,
                        'Color': palette[model_idx]
                    })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create plot
        sns.boxplot(
            x='Probability', 
            y='Value', 
            hue='N_reg',
            data=df_plot,
            ax=ax,
            palette=palette,
            width=0.6
        )
        
        ax.set_title(f'{ylabel} Comparison')
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_xlabel('Pre-training Condition', fontsize=20)
        ax.legend(title='N_reg',fontsize=20, title_fontsize=20)
        
        # Add individual data points
        sns.stripplot(
            x='Probability',
            y='Value',
            hue='N_reg',
            data=df_plot,
            ax=ax,
            dodge=True,
            palette=palette,
            alpha=0.5,
            edgecolor='gray',
            linewidth=0.5,
            jitter=True
        )
        
        # Improve layout
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.show()

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
    
    metrics = ['log_likelihood_per_obs', 'BIC', 'AIC', 'accuracy']
    metric_titles = {
        'log_likelihood_per_obs': 'Log Likelihood per Observation',
        'BIC': 'Bayesian Information Criterion',
        'AIC': 'Akaike Information Criterion',
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
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
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
            
            # Plot the normalized values
            ax.plot(block_labels, values_norm, 
                   marker='o', 
                   color=palette[j],
                   label=model)
        
        ax.set_title(metric_titles[metric])
        ax.set_xlabel('Probability Blocks')
        ax.set_ylabel('Normalized Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
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
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]#, [2,2]
    ])
model = 'inference_based' # 'glm_prob_switch', 'glm_prob_r', 'inference_based'
if model == 'inference_based':
    n_regressors = [1,2,3,4,5]
else:
    n_regressors = [2,3,4,7,10]

metrics_data = {
    n_reg: {
        'log_likelihood_per_obs': [],
        'BIC': [],
        'AIC': [],
        'accuracy': [],
        'probs_net': []
    }
    for n_reg in n_regressors
}
    
for probs_net in blocks:
    for n_reg in n_regressors:
        folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                f"d{dec_dur}_prb{probs_net[0]}{probs_net[1]}")
        if probs_net[0] == 2:
                seed_task = 13
                folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_"f"prb_task_seed_{seed_task}")
        
        glm_dir = os.path.join(folder, f'{model}_weights_{n_reg}')
        data_dir = os.path.join(folder, f'analysis_data_{model}')
        combined_glm_metrics = os.path.join(glm_dir, 'all_subjects_glm_metrics.csv')
        combined_data_file = os.path.join(data_dir, 'all_subjects_data.csv')
        combined_glm_data = os.path.join(glm_dir, 'all_subjects_glm_regressors.csv')
        if os.path.exists(combined_glm_metrics):
            df = pd.read_csv(combined_glm_metrics, low_memory=False)
            df_data = pd.read_csv(combined_glm_data,low_memory=False)
            bad_nets = filter_df_performance(df,combined_data_file)
            df = df[~df['seed'].isin(bad_nets)]
            #To just plot one point for each seed, comment to see all cross-validation cases
            df = df.groupby('seed').mean()
            # Store metrics
            metrics_data[n_reg]['log_likelihood_per_obs'].append(df['log_likelihood_per_obs'].values)
            metrics_data[n_reg]['BIC'].append(df['BIC'].values)
            metrics_data[n_reg]['AIC'].append(df['AIC'].values)
            metrics_data[n_reg]['accuracy'].append(df['accuracy'].values)
            metrics_data[n_reg]['probs_net'].append(probs_net)
        else:
            print(f"Metrics file not found: {combined_glm_metrics}")
plot_metrics_comparison(blocks, metrics_data, n_regressors)
#plot_metrics_variance(blocks, metrics_data, n_regressors)



                
