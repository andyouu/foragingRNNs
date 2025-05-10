import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

from new_forage_analysis import weights_computation

def filter_df_performance(combined_data_file):
    underperforming_nets = []
    if os.path.exists(combined_data_file):
        data = pd.read_csv(combined_data_file)
        subjects = np.unique(data['network_seed'])
        for subj in subjects:
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


def prob_r_plot_weights_comparison(n_regressors, blocks, combined_df):
    """Main plotting function with separate colors for r_plus and r_minus"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Set distinct colors for regressor types
    plus_color = '#d62728'  # Red for r_plus
    minus_color = '#1f77b4'  # Blue for r_minus
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        fig, axes = plt.subplots(1, len(blocks), figsize=(15, 5), sharey=True)
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'plus_x': [], 'plus_y': [], 
                              'minus_x': [], 'minus_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            prob_r_plot_single_block(ax, n_reg, block, combined_df, 
                            plus_color, minus_color, summary_data)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        prob_r_plot_summary(n_reg, blocks, summary_data, plus_color, minus_color)

def prob_r_plot_single_block(ax, n_reg, block, combined_df, plus_color, minus_color, summary_data):
    """Plot a single probability block with distinct regressor colors"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return
        
    # Separate regressors
    r_plus = sorted([r for r in block_df['regressor'].unique() if 'r_plus' in r])
    r_minus = sorted([r for r in block_df['regressor'].unique() if 'r_minus' in r])
    
    # Positions for plotting
    plus_pos = np.arange(len(r_plus))
    minus_pos = np.arange(len(r_plus), len(r_plus) + len(r_minus))
    
    # Plot r_plus elements
    for i, reg in enumerate(r_plus):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=plus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=plus_color, linewidth=1)
        plt.setp(bp['caps'], color=plus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Plot r_minus elements
    for i, reg in enumerate(r_minus):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[minus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=minus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=minus_color, linewidth=1)
        plt.setp(bp['caps'], color=minus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Calculate and plot averages
    plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                for reg in r_plus]
    minus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                 for reg in r_minus]
    
    # Store for summary plot
    summary_data[block]['plus_x'] = plus_pos
    summary_data[block]['plus_y'] = plus_avgs
    summary_data[block]['minus_x'] = minus_pos
    summary_data[block]['minus_y'] = minus_avgs
    
    # Plot average lines
    if len(plus_avgs) > 0:
        ax.plot(plus_pos, plus_avgs, color=plus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'r_plus ({block[0]}/{block[1]})')
    if len(minus_avgs) > 0:
        ax.plot(minus_pos, minus_avgs, color=minus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'r_minus ({block[0]}/{block[1]})')
    
    # Formatting
    ax.set_xticks(np.concatenate([plus_pos, minus_pos]))
    ax.set_xticklabels(r_plus + r_minus, rotation=45)
    ax.set_title(f"Block: {block[0]}/{block[1]}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend()

def prob_r_plot_summary(n_reg, blocks, summary_data, plus_color, minus_color):
    """Plot all average lines with fading alpha based on probability"""
    plt.figure(figsize=(6, 3))
    
    title_fontsize = 12
    label_fontsize = 11
    tick_fontsize = 10
    legend_fontsize = 10
    
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=plus_color, label='r_plus'),
        mpatches.Patch(color=minus_color, label='r_minus')
    ]
    
    # Get all unique regressor names in order
    all_regressor_names = []
    for block in blocks:
        if len(summary_data[block]['plus_x']) > 0:
            # Get the actual regressor names from the first block that has data
            block_df = combined_df[
                (combined_df['n_regressors'] == n_reg) & 
                (combined_df['probability_block'] == block[0])]
            all_regressor_names = block_df['regressor'].unique()
            break
    all_regressor_names = all_regressor_names[1:]
    # Convert array to list
    all_regressor_names = all_regressor_names.tolist()

    all_regressor_names = all_regressor_names[len(all_regressor_names)//2:] + \
        all_regressor_names[:len(all_regressor_names)//2]
    # Create position mapping for regressors

    # Plot each probability block
    x_pos = np.arange(1,n_reg)
    plt.xticks(x_pos, [str(int(x)) for x in x_pos],  # Convert to integer strings
        fontsize=tick_fontsize)
    alpha = 1
    for block in blocks:
        # Plot r_plus averages using actual positions
        if len(summary_data[block]['plus_x']) > 0:
            #x_pos = [regressor_positions[name] for name in all_regressor_names if 'r_plus' in name]
            plt.plot(x_pos, summary_data[block]['plus_y'],
                    color=plus_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot r_minus averages using actual positions
        if len(summary_data[block]['minus_x']) > 0:
            #x_pos = [regressor_positions[name] for name in all_regressor_names if 'r_minus' in name]
            plt.plot(x_pos, summary_data[block]['minus_y'],
                    color=minus_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Add to legend
        legend_handles.append(
            mpatches.Patch(color='gray', alpha=alpha, label=f'p={block[0]}-{block[1]:.2f}')
        )
        alpha -= 0.2
    
    # Set x-ticks to show regressor names
    # if all_regressor_names:
    #     plt.xticks(range(len(all_regressor_names)), all_regressor_names, 
    #              rotation=45, fontsize=tick_fontsize, ha='right')
    
    plt.yticks(fontsize=tick_fontsize)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(f'Average Weights - {n_reg-1} Regressors', fontsize=title_fontsize, pad=12)
    plt.ylabel('Coefficient Value', fontsize=label_fontsize)
    plt.xlabel('Regressor', fontsize=label_fontsize)
    
    # Create legend showing probability fade
    legend = plt.legend(handles=legend_handles,
                      loc='upper right',
                      title="Regressor Type / Probability",
                      framealpha=0.8,
                      fontsize=legend_fontsize)
    legend.get_title().set_fontsize(legend_fontsize)
    
    plt.tight_layout()
    plt.show()


def prob_switch_plot_weights_comparison(n_regressors, blocks, combined_df):
    """Main plotting function with separate colors for r_plus and r_minus"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        fig, axes = plt.subplots(1, len(blocks), figsize=(15, 5), sharey=True)
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'plus_x': [], 'plus_y': [], 
                              'minus_x': [], 'minus_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            prob_switch_plot_single_block(ax, n_reg, block, combined_df, summary_data)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        prob_switch_plot_summary(n_reg, blocks,summary_data)

def prob_switch_plot_single_block(ax, n_reg, block, combined_df, summary_data):
    """Plot a single probability block with distinct regressor colors"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return
        
    # Separate regressors
    rss_plus = sorted(
        [r for r in block_df['regressor'].unique() if 'rss_plus' in r],
        key=lambda x: int(x.replace('rss_plus', ''))
    )

    rds_plus = sorted(
        [r for r in block_df['regressor'].unique() if 'rds_plus' in r],
        key=lambda x: int(x.replace('rds_plus', ''))
    )

    rss_minus = sorted(
        [r for r in block_df['regressor'].unique() if 'rss_minus' in r],
        key=lambda x: int(x.replace('rss_minus', ''))
    )
    last_trial = sorted([r for r in block_df['regressor'].unique() if 'last_trial' in r])

    
    # Positions for plotting
    last_trial_pos = np.arange(len(last_trial))
    rss_plus_pos = np.arange(len(last_trial), len(last_trial) + len(rss_plus))
    rds_plus_pos = np.arange(len(last_trial) + len(rss_plus), 
                            len(last_trial) + len(rss_plus) + len(rds_plus))
    rss_minus_pos = np.arange(len(last_trial) + len(rss_plus) + len(rds_plus), 
                    len(last_trial) + len(rss_plus) + len(rds_plus) + len(rss_minus))
    
    # Set distinct colors for regressor types
    rss_color = '#d62728'  # Red for r_plus
    rds_color = '#ff7f0e'  # Orange for r_minus
    minus_color = '#1f77b4'  # Blue for r_minus
    neutral_color = '#7f7f7f'  # Gray for last_trial

    # Colors for each regressor type
    rss_plus_color = rss_color
    rds_plus_color = rds_color  # or use a different shade if desired
    rss_minus_color = minus_color
    last_trial_color = neutral_color  # assuming you want a neutral color for last_trial
    
    # Plot last_trial elements
    for i, reg in enumerate(last_trial):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[last_trial_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=last_trial_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=last_trial_color, linewidth=1)
        plt.setp(bp['caps'], color=last_trial_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
            
    # Plot rss_plus elements
    for i, reg in enumerate(rss_plus):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[rss_plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=rss_plus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=rss_plus_color, linewidth=1)
        plt.setp(bp['caps'], color=rss_plus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Plot rds_plus elements
    for i, reg in enumerate(rds_plus):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[rds_plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=rds_plus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=rds_plus_color, linewidth=1)
        plt.setp(bp['caps'], color=rds_plus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Plot rss_minus elements
    for i, reg in enumerate(rss_minus):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[rss_minus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=rss_minus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=rss_minus_color, linewidth=1)
        plt.setp(bp['caps'], color=rss_minus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)

    
    # Calculate and plot averages
    rss_plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                    for reg in rss_plus]
    rds_plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                    for reg in rds_plus]
    rss_minus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                     for reg in rss_minus]
    last_trial_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                      for reg in last_trial]
    
    # Store for summary plot
    summary_data[block]['rss_plus_x'] = rss_plus_pos
    summary_data[block]['rss_plus_y'] = rss_plus_avgs
    summary_data[block]['rds_plus_x'] = rds_plus_pos
    summary_data[block]['rds_plus_y'] = rds_plus_avgs
    summary_data[block]['rss_minus_x'] = rss_minus_pos
    summary_data[block]['rss_minus_y'] = rss_minus_avgs
    summary_data[block]['last_trial_x'] = last_trial_pos
    summary_data[block]['last_trial_y'] = last_trial_avgs
    
    # Plot average lines
    if len(rss_plus_avgs) > 0:
        ax.plot(rss_plus_pos, rss_plus_avgs, color=rss_plus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rss_plus ({block[0]}/{block[1]})')
    if len(rds_plus_avgs) > 0:
        ax.plot(rds_plus_pos, rds_plus_avgs, color=rds_plus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rds_plus ({block[0]}/{block[1]})')
    if len(rss_minus_avgs) > 0:
        ax.plot(rss_minus_pos, rss_minus_avgs, color=rss_minus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rss_minus ({block[0]}/{block[1]})')
    if len(last_trial_avgs) > 0:
        ax.plot(last_trial_pos, last_trial_avgs, color=last_trial_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'last_trial ({block[0]}/{block[1]})')
    
    # Formatting
    ax.set_xticks(np.concatenate([last_trial_pos,rss_plus_pos, rds_plus_pos, rss_minus_pos]))
    ax.set_xticklabels(last_trial + rss_plus + rds_plus + rss_minus, rotation=45)
    ax.set_title(f"Block: {block[0]}/{block[1]}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend()

def prob_switch_plot_summary(n_reg, blocks, summary_data):
    """Plot all average lines with fading alpha based on probability"""
    plt.figure(figsize=(8, 4))  # Slightly larger figure to accommodate more regressors
    
        # Set distinct colors for regressor types
    rss_color = '#d62728'  # Red for r_plus
    rds_color = '#ff7f0e'  # Orange for r_minus
    minus_color = '#1f77b4'  # Blue for r_minus
    neutral_color = '#7f7f7f'  # Gray for last_trial

    title_fontsize = 12
    label_fontsize = 11
    tick_fontsize = 10
    legend_fontsize = 10
    
    # Create legend handles for regressor types
    legend_handles = [
        mpatches.Patch(color=rss_color, label='rss_plus'),
        mpatches.Patch(color=rds_color, alpha=0.7, label='rds_plus'),  # Different alpha for distinction
        mpatches.Patch(color=minus_color, label='rss_minus'),
        mpatches.Patch(color=neutral_color, label='last_trial')
    ]
    
    # Get all unique regressor names in order
    all_regressor_names = []
    for block in blocks:
        if 'rss_plus_x' in summary_data[block] and len(summary_data[block]['rss_plus_x']) > 0:
            # Get the actual regressor names from the first block that has data
            block_df = combined_df[
                (combined_df['n_regressors'] == n_reg) & 
                (combined_df['probability_block'] == block[0])]
            all_regressor_names = block_df['regressor'].unique()
            break
    
    if len(all_regressor_names) > 0:
        all_regressor_names = all_regressor_names.tolist()
        # Organize regressors by type while maintaining their original order within each type
        rss_plus_names = [name for name in all_regressor_names if 'rss_plus' in name]
        rds_plus_names = [name for name in all_regressor_names if 'rds_plus' in name]
        rss_minus_names = [name for name in all_regressor_names if 'rss_minus' in name]
        last_trial_names = [name for name in all_regressor_names if 'last_trial' in name]
        
        # make the name that ents with "_10" to be the last one for all lists
        rss_plus_names = sorted(
            [name for name in all_regressor_names if 'rss_plus' in name],
            key=lambda x: int(x.replace('rss_plus', ''))  # Extract number after 'rss_plus'
        )

        rds_plus_names = sorted(
            [name for name in all_regressor_names if 'rds_plus' in name],
            key=lambda x: int(x.replace('rds_plus', ''))  # Extract number after 'rds_plus'
        )

        rss_minus_names = sorted(
            [name for name in all_regressor_names if 'rss_minus' in name],
            key=lambda x: int(x.replace('rss_minus', ''))  # Extract number after 'rss_minus'
        )
        
        # Combine them in the desired order
        all_regressor_names = last_trial_names + rss_plus_names + rds_plus_names + rss_minus_names

    # Create position mapping for regressors
    #regressor_positions = {name: i for i, name in enumerate(all_regressor_names)}

    # Plot each probability block
    alpha = 1
    x_pos = np.arange(1, n_reg+1)
    plt.xticks(x_pos, [str(int(x)) for x in x_pos],  # Convert to integer strings
    fontsize=tick_fontsize)
    for block in blocks:
        prob = block[0]
        
        # Plot rss_plus averages
        if 'rss_plus_x' in summary_data[block] and len(summary_data[block]['rss_plus_x']) > 0:
            
            plt.plot(x_pos[1:], summary_data[block]['rss_plus_y'],
                    color=rss_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot rds_plus averages
        if 'rds_plus_x' in summary_data[block] and len(summary_data[block]['rds_plus_x']) > 0:
            plt.plot(x_pos[1:], summary_data[block]['rds_plus_y'],
                    color=rds_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot rss_minus averages
        if 'rss_minus_x' in summary_data[block] and len(summary_data[block]['rss_minus_x']) > 0:
            plt.plot(x_pos[1:], summary_data[block]['rss_minus_y'],
                    color=minus_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot last_trial averages
        if 'last_trial_x' in summary_data[block] and len(summary_data[block]['last_trial_x']) > 0:
            plt.plot(x_pos[0], summary_data[block]['last_trial_y'],
                    color=neutral_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Add to legend
        legend_handles.append(
            mpatches.Patch(color='gray', alpha=alpha, label=f'p={block[0]}-{block[1]:.2f}')
        )
        alpha -= 0.2
    
    plt.yticks(fontsize=tick_fontsize)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(f'Average Weights - {n_reg-1} Regressors', fontsize=title_fontsize, pad=12)
    plt.ylabel('Coefficient Value', fontsize=label_fontsize)
    plt.xlabel('Regressor', fontsize=label_fontsize)
    
    # Create legend showing probability fade
    legend = plt.legend(handles=legend_handles,
                      loc = 'best',
                      title="Regressor Type / Probability",
                      framealpha=0.8,
                      fontsize=legend_fontsize)
    legend.get_title().set_fontsize(legend_fontsize)
    
    plt.tight_layout()
    plt.show()
def inf_based_plot_weights_comparison(n_regressors,blocks,combined_df):
    """Main plotting function with separate colors for V and s"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        fig, axes = plt.subplots(1, len(blocks), figsize=(12, 4), sharey=True)
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'v_x': [], 'v_y': [], 
                              's_x': [], 's_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            inf_based_plot_single_block(ax, n_reg, block, combined_df, summary_data)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        inf_based_plot_summary(n_reg, blocks, summary_data)

def inf_based_plot_single_block(ax, n_reg, block, combined_df, summary_data):
    """Plot a single probability block with distinct regressor colors"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return
        
    # Separate and sort regressors numerically
    v_regs = sorted([r for r in block_df['regressor'].unique() if 'V_t' in r])
    s_regs = sorted([r for r in block_df['regressor'].unique() if 'side_num' in r])
    
    # Positions for plotting
    v_pos = np.arange(len(v_regs))
    s_pos = np.arange(len(v_regs), len(v_regs) + len(s_regs))
    
    # Set distinct colors
    v_color = '#d62728'  # Red for V
    s_color = '#1f77b4'  # Blue for s
    
    # Plot V elements
    for i, reg in enumerate(v_regs):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[v_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=v_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=v_color, linewidth=1)
        plt.setp(bp['caps'], color=v_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Plot s elements
    for i, reg in enumerate(s_regs):
        data = block_df[block_df['regressor'] == reg]['coefficient']
        bp = ax.boxplot(data, positions=[s_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=s_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=s_color, linewidth=1)
        plt.setp(bp['caps'], color=s_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
    
    # Calculate and plot averages
    v_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
             for reg in v_regs]
    s_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
             for reg in s_regs]
    
    # Store for summary plot
    summary_data[block]['v_x'] = v_pos
    summary_data[block]['v_y'] = v_avgs
    summary_data[block]['s_x'] = s_pos
    summary_data[block]['s_y'] = s_avgs
    
    # Plot average lines
    if len(v_avgs) > 0:
        ax.plot(v_pos, v_avgs, color=v_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'V ({block[0]}/{block[1]})')
    if len(s_avgs) > 0:
        ax.plot(s_pos, s_avgs, color=s_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f's ({block[0]}/{block[1]})')
    
    # Formatting
    ax.set_xticks(np.concatenate([v_pos, s_pos]))
    ax.set_xticklabels(v_regs + s_regs, rotation=45)
    ax.set_title(f"Block: {block[0]}/{block[1]}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend()

def inf_based_plot_summary(n_reg, blocks, summary_data):
    """Plot all average lines with fading alpha based on probability"""
    plt.figure(figsize=(6, 4))
    
    # Set colors
    v_color = '#d62728'  # Red for V
    s_color = '#1f77b4'  # Blue for s

    title_fontsize = 12
    label_fontsize = 11
    tick_fontsize = 10
    legend_fontsize = 10
    
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=v_color, label='V'),
        mpatches.Patch(color=s_color, label='s')
    ]
    
    # Get all unique regressor names in order
    all_regressor_names = []
    for block in blocks:
        if 'v_x' in summary_data[block] and len(summary_data[block]['v_x']) > 0:
            block_df = combined_df[
                (combined_df['n_regressors'] == n_reg) & 
                (combined_df['probability_block'] == block[0])]
            all_regressor_names = block_df['regressor'].unique()
            break
    
    if len(all_regressor_names) > 0:
        all_regressor_names = all_regressor_names.tolist()
        v_names = sorted([r for r in block_df['regressor'].unique() if 'V_t' in r])
        s_names = sorted([r for r in block_df['regressor'].unique() if 'side_num' in r])
        all_regressor_names = v_names + s_names

    # Create position mapping
    regressor_positions = {name: i for i, name in enumerate(all_regressor_names)}

    # Plot each probability block
    alpha = 1
    for block in blocks:
        prob = block[0]
        
        # Plot V averages
        if 'v_x' in summary_data[block] and len(summary_data[block]['v_x']) > 0:
            x_pos = [regressor_positions[name] for name in all_regressor_names if 'V' in name]
            plt.plot(x_pos, summary_data[block]['v_y'],
                    color=v_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot s averages
        if 's_x' in summary_data[block] and len(summary_data[block]['s_x']) > 0:
            x_pos = [regressor_positions[name] for name in all_regressor_names if 's' in name]
            plt.plot(x_pos, summary_data[block]['s_y'],
                    color=s_color, linewidth=2, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Add probability to legend
        if block == blocks[0]:
            legend_handles.append(mlines.Line2D([], [], color='gray', alpha=alpha,
                                             linewidth=2, label=f'p={prob:.2f}'))
        alpha -= 0.2
    
    # Formatting
    if all_regressor_names:
        plt.xticks(range(len(all_regressor_names)), all_regressor_names, 
                 rotation=45, fontsize=tick_fontsize, ha='right')
    
    plt.yticks(fontsize=tick_fontsize)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(f'Average Weights - {n_reg} Regressors', fontsize=title_fontsize, pad=12)
    plt.ylabel('Coefficient Value', fontsize=label_fontsize)
    plt.xlabel('Regressor', fontsize=label_fontsize)
    
    legend = plt.legend(handles=legend_handles,
                      loc='best',
                      title="Regressor Type / Probability",
                      framealpha=0.8,
                      fontsize=legend_fontsize)
    legend.get_title().set_fontsize(legend_fontsize)
    
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
        #not considering the custom block fot this plots because of confusion
        [0, 0.9],[0.2, 0.8],[0.3, 0.7],[0.4, 0.6]#, [2,2]
    ])
model = 'glm_prob_switch'
if model == 'inference_based':
    n_regressors = [1,2,3,4,5]
else:
    n_regressors = [2,3,5,7,10]

all_df = []
for n_reg in n_regressors:
    for probs_net in blocks:
        folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                f"d{dec_dur}_prb{probs_net[0]}{probs_net[1]}")
        if probs_net[0] == 2:
                seed_task = 13
                folder = (f"{main_folder}/ForagingBlocks_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_"f"prb_task_seed_{seed_task}")
        
        glm_dir = os.path.join(folder, f'{model}_weights_{n_reg}')
        data_dir = os.path.join(folder, f'analysis_data_{model}')
        combined_glm_file = os.path.join(glm_dir, 'all_subjects_weights.csv')
        combined_data_file = os.path.join(data_dir, 'all_subjects_data.csv')
        combined_glm_data = os.path.join(glm_dir, 'all_subjects_glm_regressors.csv')
        if os.path.exists(combined_glm_file):
            df = pd.read_csv(combined_glm_file)
            df_data = pd.read_csv(combined_glm_data)
            df = df.groupby(['seed', 'regressor'])['coefficient'].mean().reset_index()
            bad_nets = filter_df_performance(combined_data_file)
            df = df[~df['seed'].isin(bad_nets)]
            df['probability_block'] = probs_net[0]
            df['n_regressors'] = n_reg
            all_df.append(df)
        else:
            print(f"Weights file not found: {combined_glm_file}")
combined_df = pd.concat(all_df, ignore_index=True)
if model == 'glm_prob_r':
    prob_r_plot_weights_comparison(n_regressors,blocks,combined_df)
elif model == 'glm_prob_switch':
    prob_switch_plot_weights_comparison(n_regressors,blocks,combined_df)
elif model == 'inference_based':
    inf_based_plot_weights_comparison(n_regressors,blocks,combined_df)




                
