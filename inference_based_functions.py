import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from typing import Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from scipy.optimize import minimize
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
        y = np.exp(-beta *x + s * alpha)
        return 1/(1 + y)


def softmax_prob_right(V, s, beta, T):
    """
    Softmax choice probability for right option:
    P(right | V, s) = σ(β(V + s*T)) = 1 / (1 + exp(-β(V + s*T)))
    
    Parameters:
    V : array-like
        Expected values
    s : array-like 
        Side indicators (-1 for left, 1 for right)
    beta : float
        Inverse temperature (slope)
    T : float
        Side bias scaling
    """
    z = beta * (V + s * T)
    return 1 / (1 + np.exp(-z))  # Sigmoid implementation of softmax for binary choice


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

def negative_log_likelihood(params, V, s, choices):
    """Objective function for minimization"""
    beta, T = params
    p_right = softmax_prob_right(V, s, beta, T)
    epsilon = 1e-15
    p_right = np.clip(p_right, epsilon, 1-epsilon)
    return -np.sum(choices * np.log(p_right) + (1-choices) * np.log(1-p_right))


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
    select_columns = ['reward', 'actions', 'iti', 'prob_r','split_label']
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
    new_df['choice_1'] = new_df['choice'].shift(-1)
    new_df.loc[(new_df['choice_1'] == 0), 'side_num'] = -1
    new_df.loc[(new_df['choice_1'] == 1), 'side_num'] = 1
    #return all but last bc it contanis a nan
    return new_df[:-1]

def manual_computation_v2(df: pd.DataFrame, p_SW: float, p_RWD: float) -> pd.DataFrame:
    """
    Compute V_t based on the recursive equations for R_t and the given parameters.
    
    Args:
        df: Input DataFrame containing trial data
        p_SW: Probability of switching from active to inactive state
        p_RWD: Probability of reward in active state
        p_RW0: Base probability parameter for V_t computation
        
    Returns:
        DataFrame with computed V_t values and intermediate calculations
    """
    # Select relevant columns and create a working copy
    select_columns = ['reward', 'actions', 'iti', 'prob_r', 'split_label']
    df_glm = df.loc[:, select_columns].copy()
    
    # Transform actions (originally 2=left, 3=right) to binary choice (0=left, 1=right)
    df_glm['outcome_bool'] = df_glm['reward']  # Copy reward as boolean outcome
    df_glm['choice'] = df_glm['actions'] - 2   # Convert actions to 0/1
    df_glm.loc[df_glm['choice'] < 0, 'choice'] = np.nan  # Handle invalid values
    
    # Reset index and create new working dataframe
    df = df_glm.reset_index(drop=True)
    new_df = df.copy()
    new_df = new_df.dropna(subset=['choice']).reset_index(drop=True)
    
    # Prepare columns for side and choice tracking
    new_df['choice_1'] = new_df['choice'].shift(-1)
    new_df.loc[(new_df['choice_1'] == 0), 'side_num'] = -1
    new_df.loc[(new_df['choice_1'] == 1), 'side_num'] = 1
    
    # Initialize variables for the recursive computation
    new_df['R_t'] = 0.0
    new_df['V_t'] = 0.0
    new_df['same_site'] = (new_df['choice'] == new_df['choice'].shift(1)).astype(int)
    new_df.loc[0, 'same_site'] = 0  # First trial has no previous site
    
    # Compute rho parameter
    rho = 1 / ((1 - p_SW) * (1 - p_RWD))
    
    # Iterate through trials to compute R_t and V_t
    for t in range(len(new_df)):
        if t == 0:
            # First trial starts with R_t = 0
            new_df.at[t, 'R_t'] = 0.0
        else:
            if new_df.at[t, 'reward']:
                # Reward resets R_t to 0
                new_df.at[t, 'R_t'] = 0.0
            else:
                if new_df.at[t, 'same_site']:
                    # Same site: apply the recursive equation
                    prev_R = new_df.at[t-1, 'R_t']
                    new_df.at[t, 'R_t'] = rho * (prev_R + p_SW)
                else:
                    # Different site: need to handle site switching (implementation depends on specific requirements)
                    # This is a placeholder - you may need to adjust based on how site switching affects R_t
                    prev_R = new_df.at[t-1, 'R_t']
                    #new_df.at[t, 'R_t'] = ((prev_R + p_SW) / (1 - p_SW)) * (1 / (1 - p_RWD))
                    new_df.at[t, 'R_t'] = prev_R

        
        # Compute V_t from R_t
        R_t = new_df.at[t, 'R_t']
        if R_t == 0:
            new_df.at[t, 'V_t'] = new_df.at[t, 'side_num'] * p_RWD
        elif R_t == new_df.at[t-1, 'R_t']:
            new_df.at[t, 'V_t'] = -new_df.at[t, 'side_num'] * p_RWD
        else:
            new_df.at[t, 'V_t'] = p_RWD * (1 - 2 / (R_t**(-new_df.at[t, 'side_num']) + 1))
    #plot histogram of V_t
    # new_df['V_t'].hist(bins=30)
    # plt.show()
    
    return new_df[:-1]


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


def plot_inference_prob_r(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    beta = GLM_df.loc[GLM_df['regressor'].str.contains('V_t'), 'coefficient']
    s = GLM_df.loc[GLM_df['regressor'].str.contains('s'), 'coefficient']
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(beta)], beta, marker='.', color='indianred', alpha=alpha)
    ax.plot(orders[:len(s)], s, marker='.', color='teal', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='beta'),
        mpatches.Patch(color='teal', label='T')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')

def metric_computation_optim(df,split):    
    # Create a copy to avoid modifying original dataframe
    df_train = df[df['split_label'] == f'train_{split+1}']
    df_test = df[df['split_label'] == f'test_{split+1}']
    
    # Fit logistic regression model
    try:
        mM_logit = smf.logit(formula='choice ~ V_t + side_num', data=df_train).fit()
    except Exception as e:
        print(f"Model fitting failed: {str(e)}")
        return None, None
    
    # Create coefficients DataFrame
    GLM_df = pd.DataFrame({
        'coefficient': mM_logit.params,
        'std_err': mM_logit.bse,
        'z_value': mM_logit.tvalues,  # Note: statsmodels uses z-values for logit, not t-values
        'p_value': mM_logit.pvalues,
        'conf_Interval_Low': mM_logit.conf_int()[0],
        'conf_Interval_High': mM_logit.conf_int()[1]
    })
    
    # Add predicted probabilities to dataframe
    df_test['pred_prob'] = mM_logit.predict(df_test)
    
    #plot an histogram of the predicted probabilities
    df_test['pred_prob'].hist(bins=30)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    # Prepare true values and predictions
    y_true = df_test['choice'].values  # Using all observations
    y_pred_prob = df_test['pred_prob'].values
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    # Create probabilistic predictions
    np.random.seed(42)
    y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int)
    
    # Calculate metrics
    metrics_dict = {
        # Model information
        "n_obs": len(y_true),
        "n_params": len(mM_logit.params),
        
        # Log-likelihood
        "log_likelihood": mM_logit.llf,
        "log_likelihood_per_obs": mM_logit.llf / len(y_true),
        "null_log_likelihood": mM_logit.llnull,
        
        # Information criteria
        "AIC": mM_logit.aic,
        "BIC": mM_logit.bic,
        
        # Pseudo R-squared
        "pseudo_r2_mcfadden": mM_logit.prsquared,
        "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),
        "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                            (1 - np.exp(2 * mM_logit.llnull / len(y_true))),
        
        # Classification metrics
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class, zero_division=0),
        "recall": recall_score(y_true, y_pred_class, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_class, zero_division=0),
        "accuracy_bis": accuracy_score(y_true, y_pred_class_mult),
        "precision_bis": precision_score(y_true, y_pred_class_mult, zero_division=0),
        "recall_bis": recall_score(y_true, y_pred_class_mult, zero_division=0),
        "f1_score_bis": f1_score(y_true, y_pred_class_mult, zero_division=0),
        
        # Probability metrics
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "brier_score": brier_score_loss(y_true, y_pred_prob),
    }
    regressors_string = 'V_t + side_num'
    return GLM_df,regressors_string,df, pd.DataFrame([metrics_dict])
    
    

def inference_data(df,split, n_back,v2=False):
    if v2:
        df_values_new = manual_computation_v2(df, p_SW=0.1, p_RWD=0.9)
    else:
        df_values_new = manual_computation(df,n_back)
    #to explore parameters:
    # df,GLM_metrics = metric_computation_brut_force(df_values_new)
    GLM_df, regressors_string, df_regressors, df_metrics = metric_computation_optim(df_values_new,split)
    return GLM_df, regressors_string, df_regressors, df_metrics
def inference_plot(ax,df_values_new):
    train, test = sequential_train_test_split(df_values_new, test_size=0.2)
    df_values_new = psychometric_fit(ax,[test,train])