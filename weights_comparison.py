import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

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
    """Main plotting function with separate colors for r_plus, r_minus and intercept"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Set distinct colors for regressor types
    plus_color = '#d62728'  # Red for r_plus
    minus_color = '#1f77b4'  # Blue for r_minus
    intercept_color = '#2ca02c'  # Green for intercept
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        # Create a summary table for this n_reg
        summary_table = []
        
        fig, axes = plt.subplots(1, len(blocks), figsize=(15, 5), sharey=True)
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'plus_x': [], 'plus_y': [], 
                              'minus_x': [], 'minus_y': [],
                              'intercept_x': [], 'intercept_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            block_table = prob_r_plot_single_block(ax, n_reg, block, combined_df, 
                            plus_color, minus_color, intercept_color, summary_data)
            summary_table.extend(block_table)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        prob_r_plot_summary(n_reg, blocks, summary_data, plus_color, minus_color, intercept_color)
        
        # Display summary table
        if summary_table:
            summary_df = pd.DataFrame(summary_table)
            print(f"\nSummary Statistics for {n_reg} Regressors:")
            display(summary_df)  # Using print instead of display

def prob_r_plot_single_block(ax, n_reg, block, combined_df, plus_color, minus_color, intercept_color, summary_data):
    """Plot a single probability block with distinct regressor colors and significance"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return []
        
    # Separate regressors
    r_plus = sorted([r for r in block_df['regressor'].unique() if 'r_plus' in r])
    r_minus = sorted([r for r in block_df['regressor'].unique() if 'r_minus' in r])
    intercepts = sorted([r for r in block_df['regressor'].unique() if 'intercept' in r.lower()])
    
    # Positions for plotting
    plus_pos = np.arange(len(r_plus))
    minus_pos = np.arange(len(r_plus), len(r_plus) + len(r_minus))
    intercept_pos = np.arange(len(r_plus) + len(r_minus), 
                      len(r_plus) + len(r_minus) + len(intercepts))
    
    # Store statistics for table
    block_stats = []
    
    # Plot r_plus elements
    for i, reg in enumerate(r_plus):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        
        bp = ax.boxplot(data, positions=[plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=plus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=plus_color, linewidth=1)
        plt.setp(bp['caps'], color=plus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        # Get precomputed p-value
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        
        # Add significance stars
        sig = get_significance_stars(p_val)
            
        # Add significance marker above boxplot
        y_max = np.max(data)
        ax.text(plus_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=plus_color, fontweight='bold')
        
        # Store stats for table
        block_stats.append(create_stats_entry(reg, 'r_plus', reg_df, block, p_val, sig))
    
    # Plot r_minus elements
    for i, reg in enumerate(r_minus):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        
        bp = ax.boxplot(data, positions=[minus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=minus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=minus_color, linewidth=1)
        plt.setp(bp['caps'], color=minus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        # Get precomputed p-value
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        
        # Add significance stars
        sig = get_significance_stars(p_val)
            
        # Add significance marker above boxplot
        y_max = np.max(data)
        ax.text(minus_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=minus_color, fontweight='bold')
        
        # Store stats for table
        block_stats.append(create_stats_entry(reg, 'r_minus', reg_df, block, p_val, sig))
    
    # Plot intercept elements
    for i, reg in enumerate(intercepts):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        
        bp = ax.boxplot(data, positions=[intercept_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=intercept_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=intercept_color, linewidth=1)
        plt.setp(bp['caps'], color=intercept_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        # Get precomputed p-value
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        
        # Add significance stars
        sig = get_significance_stars(p_val)
            
        # Add significance marker above boxplot
        y_max = np.max(data)
        ax.text(intercept_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=intercept_color, fontweight='bold')
        
        # Store stats for table
        block_stats.append(create_stats_entry(reg, 'intercept', reg_df, block, p_val, sig))
    
    # Calculate and plot averages
    plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                for reg in r_plus]
    minus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                 for reg in r_minus]
    intercept_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                     for reg in intercepts]
    
    # Store for summary plot
    summary_data[block]['plus_x'] = plus_pos
    summary_data[block]['plus_y'] = plus_avgs
    summary_data[block]['minus_x'] = minus_pos
    summary_data[block]['minus_y'] = minus_avgs
    summary_data[block]['intercept_x'] = intercept_pos
    summary_data[block]['intercept_y'] = intercept_avgs
    
    # Plot average lines
    if len(plus_avgs) > 0:
        ax.plot(plus_pos, plus_avgs, color=plus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'r_plus ({block[0]}/{block[1]})')
    if len(minus_avgs) > 0:
        ax.plot(minus_pos, minus_avgs, color=minus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'r_minus ({block[0]}/{block[1]})')
    if len(intercept_avgs) > 0:
        ax.plot(intercept_pos, intercept_avgs, color=intercept_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'intercept ({block[0]}/{block[1]})')
    
    # Formatting
    all_pos = np.concatenate([plus_pos, minus_pos, intercept_pos])
    all_labels = r_plus + r_minus + intercepts
    ax.set_xticks(all_pos)
    ax.set_xticklabels(all_labels, rotation=45)
    ax.set_title(f"Block: {block[0]}/{block[1]}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend()
    
    return block_stats

def get_significance_stars(p_val):
    """Helper function to get significance stars"""
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    return 'ns'

def create_stats_entry(reg, reg_type, reg_df, block, p_val, sig):
    """Helper function to create stats dictionary entry"""
    return {
        'Regressor': reg,
        'Type': reg_type,
        'Mean': reg_df['coefficient'].mean(),
        'CI_lower': reg_df['conf_Interval_Low'].mean(),
        'CI_upper': reg_df['conf_Interval_High'].mean(),
        'p-value': p_val,
        'Significance': sig,
        'Block': f"{block[0]}/{block[1]}"
    }


def prob_r_plot_summary(n_reg, blocks, summary_data, plus_color, minus_color,intercept_color):
    """Plot all average lines with fading alpha based on probability"""
    plt.figure(figsize=(46.8, 33.1))
    
    title_fontsize = 50
    label_fontsize = 50
    tick_fontsize = 30
    legend_fontsize = 25
    
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=plus_color, label='r_plus'),
        mpatches.Patch(color=minus_color, label='r_minus'),
        mpatches.Patch(color=intercept_color, label='intercept')
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
    
    # Convert array to list and exclude intercept for the main plot (we'll add it separately)
    all_regressor_names = [name for name in all_regressor_names.tolist() if 'intercept' not in name.lower()]
    all_regressor_names = all_regressor_names[len(all_regressor_names)//2:] + \
        all_regressor_names[:len(all_regressor_names)//2]

    # Plot each probability block
    x_pos = np.arange(1, n_reg)
    plt.xticks(x_pos, [str(int(x)) for x in x_pos],  # Convert to integer strings
        fontsize=tick_fontsize)
    
    alpha = 1
    for block in blocks:
        # Plot r_plus averages using actual positions
        if len(summary_data[block]['plus_y']) > 0:
            plt.plot(x_pos[:len(summary_data[block]['plus_y'])], summary_data[block]['plus_y'],
                    color=plus_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot r_minus averages using actual positions
        if len(summary_data[block]['minus_y']) > 0:
            plt.plot(x_pos[:len(summary_data[block]['minus_y'])], summary_data[block]['minus_y'],
                    color=minus_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Plot intercept if it exists (at position 0)
        if len(summary_data[block]['intercept_y']) > 0:
            plt.plot([0], [summary_data[block]['intercept_y'][0]],
                    color=intercept_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=6, markeredgecolor='white')
        
        # Add to legend
        legend_handles.append(
            mpatches.Patch(color='gray', alpha=alpha, label=f'p={block[0]}-{block[1]:.2f}')
        )
        alpha -= 0.2
    
    # Add x-tick for intercept at position 0
    if any(len(summary_data[block]['intercept_y']) > 0 for block in blocks):
        current_ticks = list(plt.xticks()[0])
        current_labels = [label.get_text() for label in plt.gca().get_xticklabels()]
        plt.xticks([0] + current_ticks, ['Intercept'] + current_labels, 
                 fontsize=tick_fontsize)
    
    plt.yticks(fontsize=tick_fontsize)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
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
    """Main plotting function with significance stars and stats table"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        # Create a summary table for this n_reg
        summary_table = []
        
        fig, axes = plt.subplots(1, len(blocks), figsize=(15, 5), sharey=True)
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'rss_plus_x': [], 'rss_plus_y': [], 
                             'rds_plus_x': [], 'rds_plus_y': [],
                             'rss_minus_x': [], 'rss_minus_y': [],
                             'last_trial_x': [], 'last_trial_y': [],
                             'intercept_x': [], 'intercept_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            block_table = prob_switch_plot_single_block(ax, n_reg, block, combined_df, summary_data)
            summary_table.extend(block_table)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        prob_switch_plot_summary(n_reg, blocks, summary_data)
        
        # Display summary table if data exists
        if summary_table:
            summary_df = pd.DataFrame(summary_table)
            print(f"\nSummary Statistics for {n_reg} Regressors:")
            print(summary_df.to_string())

def prob_switch_plot_single_block(ax, n_reg, block, combined_df, summary_data):
    """Plot a single probability block with significance markers"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return []
        
    # Separate regressors
    rss_plus = sorted(
        [r for r in block_df['regressor'].unique() if 'rss_plus' in r],
        key=lambda x: int(x.replace('rss_plus', '')))
    rds_plus = sorted(
        [r for r in block_df['regressor'].unique() if 'rds_plus' in r],
        key=lambda x: int(x.replace('rds_plus', '')))
    rss_minus = sorted(
        [r for r in block_df['regressor'].unique() if 'rss_minus' in r],
        key=lambda x: int(x.replace('rss_minus', '')))
    last_trial = sorted([r for r in block_df['regressor'].unique() if 'last_trial' in r])
    intercepts = [r for r in block_df['regressor'].unique() if 'intercept' in r.lower()]
    
    # Positions for plotting
    intercept_pos = np.array([0])
    last_trial_pos = np.arange(1, len(last_trial)+1)
    rss_plus_pos = np.arange(len(last_trial)+1, len(last_trial)+1 + len(rss_plus))
    rds_plus_pos = np.arange(len(last_trial)+1 + len(rss_plus), 
                          len(last_trial)+1 + len(rss_plus) + len(rds_plus))
    rss_minus_pos = np.arange(len(last_trial)+1 + len(rss_plus) + len(rds_plus), 
                   len(last_trial)+1 + len(rss_plus) + len(rds_plus) + len(rss_minus))
    
    # Colors
    rss_color = '#d62728'
    rds_color = '#ff7f0e'
    minus_color = '#1f77b4'
    neutral_color = '#7f7f7f'
    intercept_color = '#2ca02c'
    
    # Store statistics for table
    block_stats = []
    
    # Plot intercept elements with significance
    for i, reg in enumerate(intercepts):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=intercept_pos, widths=0.6,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor=intercept_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=intercept_color, linewidth=1)
        plt.setp(bp['caps'], color=intercept_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        # Add significance marker
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(intercept_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=intercept_color, fontweight='bold')
        
        # Add to stats table
        block_stats.append(create_stats_entry(reg, 'intercept', reg_df, block, p_val, sig))
    
    # Plot last_trial elements with significance
    for i, reg in enumerate(last_trial):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[last_trial_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor=neutral_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=neutral_color, linewidth=1)
        plt.setp(bp['caps'], color=neutral_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(last_trial_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=neutral_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 'last_trial', reg_df, block, p_val, sig))
    
    # Plot rss_plus elements with significance
    for i, reg in enumerate(rss_plus):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[rss_plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor=rss_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=rss_color, linewidth=1)
        plt.setp(bp['caps'], color=rss_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(rss_plus_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=rss_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 'rss_plus', reg_df, block, p_val, sig))
    
    # Plot rds_plus elements with significance
    for i, reg in enumerate(rds_plus):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[rds_plus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor=rds_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=rds_color, linewidth=1)
        plt.setp(bp['caps'], color=rds_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(rds_plus_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=rds_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 'rds_plus', reg_df, block, p_val, sig))
    
    # Plot rss_minus elements with significance
    for i, reg in enumerate(rss_minus):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[rss_minus_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor=minus_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=minus_color, linewidth=1)
        plt.setp(bp['caps'], color=minus_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(rss_minus_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=minus_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 'rss_minus', reg_df, block, p_val, sig))

    # Calculate and store averages for summary plot
    intercept_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                     for reg in intercepts]
    last_trial_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                      for reg in last_trial]
    rss_plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                    for reg in rss_plus]
    rds_plus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                    for reg in rds_plus]
    rss_minus_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                     for reg in rss_minus]
    
    summary_data[block]['intercept_x'] = intercept_pos
    summary_data[block]['intercept_y'] = intercept_avgs
    summary_data[block]['last_trial_x'] = last_trial_pos
    summary_data[block]['last_trial_y'] = last_trial_avgs
    summary_data[block]['rss_plus_x'] = rss_plus_pos
    summary_data[block]['rss_plus_y'] = rss_plus_avgs
    summary_data[block]['rds_plus_x'] = rds_plus_pos
    summary_data[block]['rds_plus_y'] = rds_plus_avgs
    summary_data[block]['rss_minus_x'] = rss_minus_pos
    summary_data[block]['rss_minus_y'] = rss_minus_avgs
    
    # Plot average lines
    if len(intercept_avgs) > 0:
        ax.plot(intercept_pos, intercept_avgs, color=intercept_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'intercept ({block[0]}/{block[1]})')
    if len(last_trial_avgs) > 0:
        ax.plot(last_trial_pos, last_trial_avgs, color=neutral_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'last_trial ({block[0]}/{block[1]})')
    if len(rss_plus_avgs) > 0:
        ax.plot(rss_plus_pos, rss_plus_avgs, color=rss_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rss_plus ({block[0]}/{block[1]})')
    if len(rds_plus_avgs) > 0:
        ax.plot(rds_plus_pos, rds_plus_avgs, color=rds_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rds_plus ({block[0]}/{block[1]})')
    if len(rss_minus_avgs) > 0:
        ax.plot(rss_minus_pos, rss_minus_avgs, color=minus_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'rss_minus ({block[0]}/{block[1]})')
    
    # Formatting
    all_pos = np.concatenate([intercept_pos, last_trial_pos, rss_plus_pos, rds_plus_pos, rss_minus_pos])
    all_labels = intercepts + last_trial + rss_plus + rds_plus + rss_minus
    ax.set_xticks(all_pos)
    ax.set_xticklabels(all_labels, rotation=45)
    ax.set_title(f"Block: {block[0]}/{block[1]}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend()
    
    return block_stats

def prob_switch_plot_summary(n_reg, blocks, summary_data):
    """Plot all average lines with fading alpha based on probability"""
    plt.figure(figsize=(46.8, 33.1))
    
    # Colors
    rss_color = '#d62728'
    rds_color = '#ff7f0e'
    minus_color = '#1f77b4'
    neutral_color = '#7f7f7f'
    intercept_color = '#2ca02c'

    title_fontsize = 50
    label_fontsize = 50
    tick_fontsize = 30
    legend_fontsize = 25
    
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=intercept_color, label='intercept'),
        mpatches.Patch(color=neutral_color, label='last_trial'),
        mpatches.Patch(color=rss_color, label='rss_plus'),
        mpatches.Patch(color=rds_color, label='rds_plus'),
        mpatches.Patch(color=minus_color, label='rss_minus')
    ]
    
    # Plot each probability block
    alpha = 1
    x_pos = np.arange(0, n_reg+2)  # Starting at 0 for intercept
    
    for block in blocks:
        # Plot intercept average
        if len(summary_data[block]['intercept_y']) > 0:
            plt.plot([0], summary_data[block]['intercept_y'],
                    color=intercept_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=9, markeredgecolor='white')
        
        # Plot last_trial average (position 1)
        if len(summary_data[block]['last_trial_y']) > 0:
            plt.plot([1], summary_data[block]['last_trial_y'],
                    color=neutral_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=9, markeredgecolor='white')
        
        # Plot rss_plus averages
        if len(summary_data[block]['rss_plus_y']) > 0:
            start_pos = 2
            end_pos = start_pos + len(summary_data[block]['rss_plus_y'])
            plt.plot(x_pos[start_pos:end_pos], summary_data[block]['rss_plus_y'],
                    color=rss_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=9, markeredgecolor='white')
        
        # Plot rds_plus averages
        if len(summary_data[block]['rds_plus_y']) > 0:
            start_pos = 2
            end_pos = start_pos + len(summary_data[block]['rds_plus_y'])
            plt.plot(x_pos[start_pos:end_pos], summary_data[block]['rds_plus_y'],
                    color=rds_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=9, markeredgecolor='white')
        
        # Plot rss_minus averages
        if len(summary_data[block]['rss_minus_y']) > 0:
            start_pos = 2
            end_pos = start_pos + len(summary_data[block]['rds_plus_y'])
            plt.plot(x_pos[start_pos:end_pos], summary_data[block]['rss_minus_y'],
                    color=minus_color, linewidth=8, alpha=alpha,
                    marker='o', markersize=9, markeredgecolor='white')
        
        # Add to legend
        legend_handles.append(
            mpatches.Patch(color='gray', alpha=alpha, label=f'p={block[0]}-{block[1]:.2f}')
        )
        alpha -= 0.2
    
    # Set x-ticks
    plt.xticks(x_pos, ['Intercept'] + [str(i) for i in range(1, n_reg+2)], 
              fontsize=tick_fontsize)
    
    plt.yticks(fontsize=tick_fontsize)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.title(f'Average Weights', fontsize=title_fontsize, pad=12)
    plt.ylabel('Coefficient Value', fontsize=label_fontsize)
    plt.xlabel('Regressor', fontsize=label_fontsize)
    
    # Create legend
    legend = plt.legend(handles=legend_handles,
                      loc='lower right',
                      title="Regressor Type / Probability",
                      framealpha=0.8,
                      fontsize=legend_fontsize)
    legend.get_title().set_fontsize(legend_fontsize)
    
    plt.tight_layout()
    plt.show()

def inf_based_plot_weights_comparison(n_regressors, blocks, combined_df):
    """Main plotting function now with intercept, significance stars, and stats table"""
    # Convert blocks to tuples if they're numpy arrays
    blocks = [tuple(block) for block in blocks]
    
    # Create individual plots for each n_regressors
    for n_reg in n_regressors:
        # Create a summary table for this n_reg
        summary_table = []
        
        fig, axes = plt.subplots(1, len(blocks), figsize=(14, 4), sharey=True)  # Slightly wider to accommodate intercept
        if len(blocks) == 1:
            axes = [axes]
        
        # Store data for summary plot
        summary_data = {block: {'intercept_x': [], 'intercept_y': [],
                              'v_x': [], 'v_y': [], 
                              's_x': [], 's_y': []} for block in blocks}
        
        for ax, block in zip(axes, blocks):
            block_table = inf_based_plot_single_block(ax, n_reg, block, combined_df, summary_data)
            summary_table.extend(block_table)
        
        fig.suptitle(f'GLM Weights Comparison - {n_reg} Regressors', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # Create summary plot for this n_reg
        inf_based_plot_summary(n_reg, blocks, summary_data,combined_df)
        
        # Display summary table if data exists
        if summary_table:
            summary_df = pd.DataFrame(summary_table)
            print(f"\nSummary Statistics for {n_reg} Regressors:")
            print(summary_df.to_string())

def inf_based_plot_single_block(ax, n_reg, block, combined_df, summary_data):
    """Plot a single block with intercept, V, and s regressors with significance"""
    block_df = combined_df[
        (combined_df['n_regressors'] == n_reg) & 
        (combined_df['probability_block'] == block[0])]
    
    if block_df.empty:
        return []
        
    # Separate and sort all regressor types
    intercepts = [r for r in block_df['regressor'].unique() if 'intercept' in r.lower()]
    v_regs = sorted([r for r in block_df['regressor'].unique() if 'V_t' in r])
    s_regs = sorted([r for r in block_df['regressor'].unique() if 'side_num' in r])
    
    # Positions for plotting (intercept at 0, then V, then s)
    intercept_pos = np.array([0])
    v_pos = np.arange(1, len(v_regs)+1)
    s_pos = np.arange(len(v_regs)+1, len(v_regs)+1 + len(s_regs))
    
    # Set distinct colors
    intercept_color = '#2ca02c'  # Green for intercept
    v_color = '#d62728'  # Red for V
    s_color = '#1f77b4'  # Blue for s
    
    # Store statistics for table
    block_stats = []
    
    # Plot intercept elements with significance
    for i, reg in enumerate(intercepts):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=intercept_pos, widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=intercept_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=intercept_color, linewidth=1)
        plt.setp(bp['caps'], color=intercept_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        # Add significance marker
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(intercept_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=intercept_color, fontweight='bold')
        
        # Add to stats table
        block_stats.append(create_stats_entry(reg, 'intercept', reg_df, block, p_val, sig))
    
    # Plot V elements with significance (unchanged except position)
    for i, reg in enumerate(v_regs):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[v_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=v_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=v_color, linewidth=1)
        plt.setp(bp['caps'], color=v_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(v_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=v_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 'V', reg_df, block, p_val, sig))
    
    # Plot s elements with significance (unchanged except position)
    for i, reg in enumerate(s_regs):
        reg_df = block_df[block_df['regressor'] == reg]
        data = reg_df['coefficient'].values
        bp = ax.boxplot(data, positions=[s_pos[i]], widths=0.6,
                       patch_artist=True, showfliers=False)
        
        plt.setp(bp['boxes'], facecolor=s_color, alpha=0.4)
        plt.setp(bp['whiskers'], color=s_color, linewidth=1)
        plt.setp(bp['caps'], color=s_color, linewidth=1)
        plt.setp(bp['medians'], color='white', linewidth=1.5)
        
        p_val = reg_df['p_value'].mean()
        p_val = scipy.stats.combine_pvalues(reg_df['p_value'], method='stouffer')[1]
        sig = get_significance_stars(p_val)
        y_max = np.max(data)
        ax.text(s_pos[i], y_max + 0.1*np.abs(y_max), sig, 
               ha='center', va='bottom', color=s_color, fontweight='bold')
        
        block_stats.append(create_stats_entry(reg, 's', reg_df, block, p_val, sig))

    # Calculate and store averages for summary plot
    intercept_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
                     for reg in intercepts]
    v_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
             for reg in v_regs]
    s_avgs = [block_df[block_df['regressor'] == reg]['coefficient'].mean() 
             for reg in s_regs]
    
    summary_data[block]['intercept_x'] = intercept_pos
    summary_data[block]['intercept_y'] = intercept_avgs
    summary_data[block]['v_x'] = v_pos
    summary_data[block]['v_y'] = v_avgs
    summary_data[block]['s_x'] = s_pos
    summary_data[block]['s_y'] = s_avgs
    
    # Plot average lines
    if len(intercept_avgs) > 0:
        ax.plot(intercept_pos, intercept_avgs, color=intercept_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'intercept ({block[0]}/{block[1]})')
    if len(v_avgs) > 0:
        ax.plot(v_pos, v_avgs, color=v_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f'V ({block[0]}/{block[1]})')
    if len(s_avgs) > 0:
        ax.plot(s_pos, s_avgs, color=s_color, linewidth=2,
               marker='o', markersize=6, markeredgecolor='white',
               label=f's ({block[0]}/{block[1]})')
    
    # Formatting
    all_pos = np.concatenate([intercept_pos, v_pos, s_pos])
    all_labels = intercepts + v_regs + s_regs
    ax.set_xticks(all_pos)
    ax.set_xticklabels(all_labels, rotation=45, fontsize=10)
    ax.set_title(f"Block: {block[0]}/{block[1]}", fontsize=20)
    ax.axhline(0, color='gray', linestyle=':')
    ax.legend(fontsize=10)
    
    return block_stats

def inf_based_plot_summary(n_reg, blocks, summary_data, combined_df):
    """Summary plot including intercept with significance"""
    # Set colors
    intercept_color = '#2ca02c'  # Green for intercept
    beta_color = '#d62728'      # Red for β (V_t)
    side_color = '#1f77b4'      # Blue for side bias
    
    # Set global styling for poster
    plt.rcParams.update({
        'axes.titlesize': 50,
        'axes.labelsize': 50,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 25,
        'lines.linewidth': 4
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(46.8, 33.1))
    
    # Get probability blocks from summary data
    prob_blocks = sorted(summary_data.keys())
    n_probs = len(prob_blocks)
    
    # Create alpha levels (darker for higher probabilities)
    alphas = np.linspace(0, 0.6, n_probs)
    
    # Plot each probability block's coefficients
    for i, block in enumerate(prob_blocks):
        # Get coefficients from summary data
        intercept = np.mean(summary_data[block]['intercept_y']) if len(summary_data[block]['intercept_y']) > 0 else 0
        beta = np.mean(summary_data[block]['v_y']) if len(summary_data[block]['v_y']) > 0 else 0
        side = np.mean(summary_data[block]['s_y']) if len(summary_data[block]['s_y']) > 0 else 0
        
        # Get p-values (averaged across regressors of each type)
        block_mask = (combined_df['n_regressors'] == n_reg) & (combined_df['probability_block'] == block[0])
        
        intercept_p = scipy.stats.combine_pvalues(combined_df[block_mask & 
                                        combined_df['regressor'].str.contains('intercept', case=False)]['p_value'], method='stouffer')[1]
        beta_p = scipy.stats.combine_pvalues(combined_df[block_mask & 
                                   combined_df['regressor'].str.contains('V_t')]['p_value'], method='stouffer')[1]
        side_p = scipy.stats.combine_pvalues(combined_df[block_mask & 
                                   combined_df['regressor'].str.contains('side_num')]['p_value'], method='stouffer')[1]
        
        # Plot bars with probability-specific alpha
        bar_width = 0.2
        bar_positions = [i-0.3, i-0.1, i+0.1]
        
        intercept_bar = ax.bar(bar_positions[0], intercept, width=bar_width, 
                             color=intercept_color, alpha=1-alphas[i],
                             edgecolor='black', linewidth=2)
        beta_bar = ax.bar(bar_positions[1], beta, width=bar_width,
                         color=beta_color, alpha=1-alphas[i],
                         edgecolor='black', linewidth=2)
        side_bar = ax.bar(bar_positions[2], side, width=bar_width,
                        color=side_color, alpha=1-alphas[i],
                        edgecolor='black', linewidth=2)
        
        # Add significance markers
        for pos, val, p_val, color in zip(bar_positions,
                                         [intercept, beta, side],
                                         [intercept_p, beta_p, side_p],
                                         [intercept_color, beta_color, side_color]):
            if not np.isnan(val) and val != 0:  # Only add if valid value
                sig = get_significance_stars(p_val)
                y_pos = val + 0.02 * abs(val) if val > 0 else val - 0.02 * abs(val)
                ax.text(pos, y_pos, sig, 
                       ha='center', va='bottom' if val > 0 else 'top', 
                       color=color, fontweight='bold', fontsize=25)
    
    # Add reference line and styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=3)
    ax.set_title('Probability-Based Inference Weights', pad=20)
    ax.set_ylabel('Coefficient Value', labelpad=20)
    
    # Set x-ticks as probability values
    prob_labels = [f'{block[0]:.2f}/{block[1]:.2f}' for block in prob_blocks]
    ax.set_xticks(np.arange(n_probs))
    ax.set_xticklabels(prob_labels, fontsize=30)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    
    # Create simplified legends
    coeff_handles = [
        mpatches.Patch(color=intercept_color, label='Intercept'),
        mpatches.Patch(color=beta_color, label='β (Value)'),
        mpatches.Patch(color=side_color, label='Side Bias')
    ]
    legend1 = ax.legend(handles=coeff_handles, title='Coefficient Types',
                      loc='upper right', bbox_to_anchor=(1.01, 1))
    
    # Add probability legend
    prob_handles = [mpatches.Patch(color='gray', alpha=1-alphas[i], 
                                  label=f'p={block[0]:.2f}/{block[1]:.2f}') 
                   for i, block in enumerate(prob_blocks)]
    legend2 = ax.legend(handles=prob_handles, title='Probability Blocks',
                      loc='upper right', bbox_to_anchor=(1.01, 0.7))
    
    # Add the first legend back
    ax.add_artist(legend1)
    
    # Adjust layout
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(right=0.75, bottom=0.2)
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
model = 'glm_prob_switch'  # or 'glm_prob_r' or 'inference_based'
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
            df = pd.read_csv(combined_glm_file,low_memory=False)
            df_data = pd.read_csv(combined_glm_data, low_memory=False)
            df_median = df.groupby(['seed', 'regressor'])['p_value'].median().reset_index()    
            df = df.groupby(['seed', 'regressor'])[['coefficient','std_err','z_value','p_value','conf_Interval_Low','conf_Interval_High']].mean().reset_index()
            df['p_value'] = df_median['p_value']
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
elif model in ['inference_based', 'inference_based_v2']:
    inf_based_plot_weights_comparison(n_regressors,blocks,combined_df)




                
