import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os



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
    label_fontsize = 24
    legend_fontsize = 20
    tick_fontsize = 18
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
    
    # Add chance level line
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, 
               linewidth=3, label='Chance Level')
    
    # Customize plot appearance with padding
    plt.title('Average Probability Right Across Probability Blocks', 
             fontsize=title_fontsize, pad=25)  # Increased pad
    plt.xlabel('Block Probability of reward (Right)', 
              fontsize=label_fontsize, labelpad=20)  # Increased labelpad
    plt.ylabel('Average probability of going right', 
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
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(pad=4.0)  # Increased padding
    
    # Save with extra padding
    plt.savefig('performance_psychophysics_poster.png', 
               dpi=300, 
               bbox_inches='tight',
               pad_inches=1)  # Added pad_inches
    
    plt.show()


def plot_metrics_comparison(blocks, metrics_data, model_names):
    """
    Create separate plots for each metric comparing models and probability conditions
    
    Args:
        blocks: Array of probability blocks (e.g., [[0.2,0.8], [0.3,0.7]])
        metrics_data: Dictionary containing metrics for each model
        model_names: List of model names (e.g., ['glm_prob_switch', 'glm_prob_r'])
    """
    # Create a consistent color palette
    palette = sns.color_palette("husl", len(model_names))
    
    # Convert blocks to readable labels
    block_labels = [f"{p[0]}/{p[1]}" for p in blocks]
    
    # Plot each metric separately
    metrics = ['log_likelihood_per_obs', 'BIC', 'AIC', 'accuracy']
    y_labels = ['Log Likelihood per Obs', 'BIC', 'AIC', 'Accuracy']
    
    # Set up figure for A0 poster - larger size and square subplots
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))  # 36 inches wide (A0 width is ~33.1 inches)
    axes = axes.flatten()
    
    # Set font sizes for poster
    title_fontsize = 24
    label_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 16
    
    for i, (metric, ylabel) in enumerate(zip(metrics, y_labels)):
        # Skip AIC for now
        if metric == 'accuracy':
            i -= 1
        if metric != 'AIC':
            ax = axes[i]
            
            # Prepare data for this metric
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
            
            # Create plot with larger elements
            sns.boxplot(
                x='Probability', 
                y='Value', 
                hue='Model',
                data=df_plot,
                ax=ax,
                palette=palette,
                width=0.6,
                linewidth=2.5  # thicker boxplot lines
            )
            
            ax.set_title(f'{ylabel} Comparison', fontsize=title_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.set_xlabel('Probability Condition (Left/Right)', fontsize=label_fontsize)
            
            # Customize legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, 
                labels, 
                title='Model',
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                bbox_to_anchor=(1.05, 1),  # Move legend outside
                loc='upper left'
            )
            
            # Add individual data points with larger markers
            sns.stripplot(
                x='Probability',
                y='Value',
                hue='Model',
                data=df_plot,
                ax=ax,
                dodge=True,
                palette=palette,
                alpha=0.5,
                edgecolor='gray',
                linewidth=1,
                jitter=True,
                size=8  # larger dots
            )
            
            # Customize tick labels
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            
            # Improve layout
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)
            
            # Make plot square
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        
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

if __name__ == '__main__':
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 38
    n_regressors = 10
    n_back = 3
    blocks = np.array([
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]#,[2,2]
    ])
    
    # Store metrics for each model
    metrics_data = {
        'glm_prob_switch': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        },
        'glm_prob_r': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        },
        'inference_based': {
            'log_likelihood_per_obs': [],
            'BIC': [],
            'AIC': [],
            'accuracy': []
        }
    }
    
    for model in ['glm_prob_switch', 'glm_prob_r', 'inference_based']:
        data_file_total = []
        for probs_net in blocks:
            folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                     f"d{dec_dur}_prb{probs_net[0]}{probs_net[1]}")
            #easy way to include custom-trained nets
            if probs_net[0] == 2:
                seed_task = 13
                folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_"f"prb_task_seed_{seed_task}")
            if(model == 'inference_based'):
                glm_dir = os.path.join(folder, f'{model}_weights_{n_back}')
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
    plot_perf_psychos(combined_data)
    plot_metrics_comparison(blocks, metrics_data, ['glm_prob_switch', 'glm_prob_r', 'inference_based'])
    plot_metrics_variance(blocks, metrics_data, ['glm_prob_switch', 'glm_prob_r', 'inference_based'])



