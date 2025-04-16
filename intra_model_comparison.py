import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from new_forage_analysis import weights_computation
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
    metrics = ['log_likelihood_per_obs', 'BIC', 'AIC', 'accuracy']
    y_labels = ['Log Likelihood per Obs', 'BIC', 'AIC', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
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
                        'N_reg': n_reg,
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
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Probability Condition (Left/Right)')
        ax.legend(title='N_reg')
        
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

if __name__ == '__main__':
    main_folder = '/home/marcaf/TFM(IDIBAPS)/rrns2/networks'
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 38
    n_regressors = 10
    
    blocks = np.array([
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]
    ])
    metrics_template = {
        'log_likelihood_per_obs': [],
        'BIC': [],
        'AIC': [],
        'accuracy': []
    }
model = 'inference_based'
if model == 'inference_based':
    n_regressors = [1,2,3,4,5]
    metrics_data = {key: metrics_template.copy() for key in n_regressors}
        
    for probs_net in blocks:
        for n_reg in n_regressors:
            folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_prb{probs_net[0]}{probs_net[1]}")
            
            glm_dir = os.path.join(folder, f'{model}_weights_{n_reg}')
            combined_glm_metrics = os.path.join(glm_dir, 'all_subjects_glm_metrics.csv')
            
            if os.path.exists(combined_glm_metrics):
                df = pd.read_csv(combined_glm_metrics)
                
                # Store metrics
                metrics_data[n_reg]['log_likelihood_per_obs'].append(df['log_likelihood_per_obs'].values)
                metrics_data[n_reg]['BIC'].append(df['BIC'].values)
                metrics_data[n_reg]['AIC'].append(df['AIC'].values)
                metrics_data[n_reg]['accuracy'].append(df['accuracy'].values)
            else:
                print(f"Metrics file not found: {combined_glm_metrics}")
                # Append empty arrays if data is missing
                metrics_data[n_reg]['log_likelihood_per_obs'].append(np.array([]))
                metrics_data[n_reg]['BIC'].append(np.array([]))
                metrics_data[n_reg]['AIC'].append(np.array([]))
                metrics_data[n_reg]['accuracy'].append(np.array([]))
    plot_metrics_comparison(blocks, metrics_data, n_regressors)



                
