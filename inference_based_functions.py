import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from typing import Tuple
from sklearn.metrics import accuracy_score
from statsmodels.formula.api import probit as sm_probit


from scipy.optimize import curve_fit


def probit(X, beta,alpha):
        """
        Return probit function with parameters alpha and beta.

        Parameters
        ----------
        x : float
            independent variable.
        beta : float
            sensitivity term. Sensitivity term corresponds to the slope of the psychometric curve.
        alpha : TYPE
            bias term. Bias term corresponds to the shift of the psychometric curve along the x-axis.

        Returns
        -------
        probit : float
            probit value for the given x, beta and alpha.

        """
        [x,s] = X
        y = np.exp(-beta * x + s * alpha)
        return 1/(1 + y)


def sequential_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = None,
    gap: int = 0,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and test sets while preserving sequential order.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with sequential data
    test_size : float, optional (default=0.2)
        Proportion of dataset to include in test split (0.0 to 1.0)
    date_col : str, optional
        Column name to use for sorting if data isn't already ordered
    gap : int, optional (default=0)
        Number of samples to leave out between train and test to avoid leakage
    random_state : int, optional
        Seed for random operations (only used if shuffling within train/test)
        
    Returns:
    --------
    train_df : pd.DataFrame
        Training set (earlier portion)
    test_df : pd.DataFrame
        Test set (later portion)
    """
    # Make a copy to avoid modifying original DataFrame
    df = df.copy()
    
    # Sort by date column if provided
    if date_col is not None and date_col in df.columns:
        df = df.sort_values(date_col)
    
    # Calculate split index
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    split_idx = n_samples - n_test - gap
    
    # Split the data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx + gap:]
    
    return train_df, test_df




def manual_computation(df: pd.DataFrame, n_back: int) -> pd.DataFrame:
    """
    Processes mouse choice-reward data to compute behavioral statistics.
    
    Args:
        df: Input DataFrame containing raw behavioral data
        n_back: Number of previous trials to consider in sequence analysis
        
    Returns:
        Processed DataFrame with computed behavioral metrics
    """
    
    # Select relevant columns and create a working copy
    select_columns = ['reward', 'actions', 'iti', 'prob_r']
    df_glm = df.loc[:, select_columns].copy()
    
    # Transform actions (originally 2=left, 3=right) to binary choice (0=left, 1=right)
    df_glm['outcome_bool'] = df_glm['reward']  # Copy reward as boolean outcome
    df_glm['choice'] = df_glm['actions'] - 2   # Convert actions to 0/1
    df_glm.loc[df_glm['choice'] < 0, 'choice'] = np.nan  # Handle invalid values
    
    # Reset index and create new working dataframe
    df = df_glm.reset_index(drop=True)
    new_df = df.copy()
    
    # Create choice-reward combination codes:
    # First digit = choice (0=left, 1=right), second digit = reward (0=no, 1=yes)
    new_df['choice_1'] = new_df['choice'].shift(1)
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 0), 'choice_rwd'] = '00'
    new_df.loc[(new_df['outcome_bool'] == 0) & (new_df['choice'] == 1), 'choice_rwd'] = '01'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 0), 'choice_rwd'] = '10'
    new_df.loc[(new_df['outcome_bool'] == 1) & (new_df['choice'] == 1), 'choice_rwd'] = '11'
    new_df['choice_rwd'] = new_df['choice_rwd'].fillna(' ')  # Fill missing with space
    
    # Clean data - remove rows with missing probability values
    new_df = new_df.dropna(subset=['prob_r']).reset_index(drop=True)
    
    # Initialize sequence analysis
    new_df['sequence'] = ''  # Will store sequence of past choice-reward pairs
    
    # Create shifted columns for n_back previous trials and build sequence string
    for i in range(n_back):
        new_df[f'choice_rwd{i+1}'] = new_df['choice_rwd'].shift(i+1)
        new_df['sequence'] = new_df['sequence'] + new_df[f'choice_rwd{i+1}']
    
    # Remove rows with incomplete sequences
    new_df = new_df.dropna(subset=['sequence']).reset_index(drop=True)
    
    # Create active side indicators based on probability
    new_df['right_active'] = 0  # 1 if right side is more probable
    new_df['left_active'] = 0   # 1 if left side is more probable
    new_df.loc[(new_df['prob_r'] > 0.5), 'right_active'] = 1
    new_df.loc[(new_df['prob_r'] < 0.5), 'left_active'] = 1
    
    # Shift active indicators to align with next trial's outcome
    # new_df['right_outcome'] = new_df['right_active'].shift(-1)
    # new_df['left_outcome'] = new_df['left_active'].shift(-1)
    new_df['right_outcome'] = new_df['right_active']
    new_df['left_outcome'] = new_df['left_active']
    
    # Calculate probability of right/left outcomes for each sequence pattern
    new_df['prob_right'] = new_df.groupby('sequence')['right_outcome'].transform('mean')
    new_df['prob_left'] = new_df.groupby('sequence')['left_outcome'].transform('mean')
    
    # Compute value difference between right and left options
    new_df['V_t'] = (new_df['prob_right'] - new_df['prob_left'])

    new_df = new_df.dropna(subset=['choice']).reset_index(drop=True)
    new_df.loc[(new_df['choice'] == 0), 'side_num'] = -1
    new_df.loc[(new_df['choice'] == 1), 'side_num'] = 1
    return new_df


def psychometric_fit(ax,df_glm_mice):
    n_bins = 10
    phi= 1
    df_80, df_20 = df_glm_mice
    bins = np.linspace(df_80['V_t'].min(), df_80['V_t'].max(), n_bins)
    df_80['binned_ev'] = pd.cut(df_80['V_t'], bins=bins)
    histogram = 0
    if histogram:
        bin_counts = df_80['binned_ev'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        bin_counts.plot(kind='bar', width=0.8, color='skyblue', edgecolor='black')
        plt.title('Histogram of Elements in Each Bin', fontsize=16)
        plt.xlabel('Bin Interval', fontsize=14)
        plt.ylabel('Number of Elements', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    grouped = df_80.groupby('binned_ev').agg(
    ev_mean=('V_t', 'mean'),
    side = ('side_num','mean'),
    p_right_mean=('choice', 'mean'),
    ).dropna() 
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    side = grouped['side'].values
    df_20['binned_ev_20'] = pd.qcut(df_20['V_t'], n_bins,duplicates='drop')
    grouped_20 = df_20.groupby('binned_ev_20').agg(
    ev_mean_20 =('V_t', 'mean'),
    p_right_mean_20=('choice', 'mean'),
    side_20=('side_num', 'mean')
    ).dropna() 
    ev_means_20 = grouped_20['ev_mean_20'].values
    p_right_mean_20 = grouped_20['p_right_mean_20'].values
    side_20 = grouped_20['side_20'].values
    bin_sizes = df_20['binned_ev_20'].value_counts(sort=False)
    [beta, alpha],_ = curve_fit(probit, [ev_means,side], p_right_mean, p0=[0, 1])
    ax.plot(ev_means_20, probit([ev_means_20,side_20], beta,alpha), color='green', label = 'Model', alpha = phi)
    #ax.plot(ev_means, psychometric(ev_means), color='grey', alpha = 0.5)
    ax.plot(ev_means_20, p_right_mean_20, marker = 'o', color = 'black',label = 'Data', alpha = phi)


def metric_computation(df):
    # Extract variables
    ev_means = df['V_t']
    side = df['side_num']
    p_right_mean = df['choice']
    true_choices = df['choice'].values  # Actual choices (0/1 or True/False)
    
    # Fit probit model
    n_bins = 20
    bins = np.linspace(df['V_t'].min(), df['V_t'].max(), n_bins)
    df['binned_ev'] = pd.cut(df['V_t'], bins=bins)
    grouped = df.groupby('binned_ev').agg(
    ev_mean=('V_t', 'mean'),
    side = ('side_num','mean'),
    p_right_mean=('choice', 'mean'),
    ).dropna() 
    ev_means_g = grouped['ev_mean'].values
    p_right_mean_g = grouped['p_right_mean'].values
    side_g = grouped['side'].values
    [beta, alpha],_ = curve_fit(probit, [ev_means_g,side_g], p_right_mean_g, p0=[0, 1])
    #this cannot be fitted with all the data
    #[beta, alpha], _ = curve_fit(probit, [ev_means, side], p_right_mean, p0=[0, 1])
    df['pred_choice'] = probit([ev_means, side], beta, alpha)
    
    # Convert probabilities to predicted choices (binary)
    predicted_choices = (df['pred_choice'] >= 0.5).astype(int)
    
    # 1. Calculate Log-Likelihood
    epsilon = 1e-15  # Small value to avoid log(0)
    ll = np.sum(
        true_choices * np.log(df['pred_choice'] + epsilon) + 
        (1 - true_choices) * np.log(1 - df['pred_choice'] + epsilon)
    )
    
    # 2. Calculate AIC
    k = 2  # Number of parameters (beta, alpha)
    aic = 2 * k - 2 * ll
    
    # 3. Calculate BIC
    n = len(df)  # Number of observations
    bic = k * np.log(n) - 2 * ll
    
    # 4. Calculate Accuracy
    accuracy = accuracy_score(true_choices, predicted_choices)
    
    # Store results
    metrics_dict = {
        "log_likelihood": ll,
        "log_likelihood_per_obs": ll/ len(true_choices),
        "accuracy": accuracy, 
        "AIC": aic,
        "BIC": bic,
    }
    GLM_metrics = pd.DataFrame([metrics_dict]) 
    return df, GLM_metrics
    

def inference_data(df, n_back):
    df_values_new = manual_computation(df,n_back)
    df,GLM_metrics = metric_computation(df_values_new)
    return df, GLM_metrics
def inference_plot(ax,df_values_new):
    train, test = sequential_train_test_split(df_values_new, test_size=0.2)
    df_values_new = psychometric_fit(ax,[test,train])