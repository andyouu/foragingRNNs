import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)



import forage_training as ft

def calculate_vif(df_glm):
    """
    Calculate Variance Inflation Factors (VIF) for a DataFrame of regressors.
    
    Args:
        df_glm (pd.DataFrame): DataFrame containing only the numeric regressors
        
    Returns:
        pd.DataFrame: VIF results with regressor names and VIF values
        
    Note:
        - Drops NaN values before calculation
        - Handles perfect collinearity cases
        - Returns None if VIF cannot be calculated
    """
    
    # Create copy and drop rows with NaN values (required for VIF calculation)
    df_temp = df_glm.select_dtypes(include=[np.number]).dropna()
    
    # Check if matrix is invertible (avoid perfect collinearity)
    if np.linalg.cond(df_temp) > 1/sys.float_info.epsilon:
        print("Warning: Matrix is singular (perfect collinearity detected)")
        return None
    
    # Calculate VIF for each regressor
    vif_data = pd.DataFrame()
    vif_data["regressor"] = df_temp.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_temp.values, i) 
        for i in range(df_temp.shape[1])
    ]
    
    # Add interpretation
    vif_data["collinearity"] = np.where(
        vif_data["VIF"] > 10, "High",
        np.where(vif_data["VIF"] > 5, "Moderate", "Low")
    )
    
    return vif_data


def psychometric(x):
    y = np.exp(-x)
    return 1/(1 + y)

def psychometric_plot(ax,df_glm_mice, data_label):
    n_bins = 10
    #equiespaced bins
    bins = np.linspace(df_glm_mice['evidence'].min(), df_glm_mice['evidence'].max(), n_bins)
    df_glm_mice['binned_ev'] = pd.cut(df_glm_mice['evidence'], bins=bins)
    #equipopulated bins
    #df_glm_mice['binned_ev'] = pd.qcut(df_glm_mice['evidence'], n_bins,duplicates='drop')
    #bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
    #print histograms
    histogram = 0
    if histogram:
        bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        bin_counts.plot(kind='bar', width=0.8, color='skyblue', edgecolor='black')
        plt.title('Histogram of Elements in Each Bin', fontsize=16)
        plt.xlabel('Bin Interval', fontsize=14)
        plt.ylabel('Number of Elements', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    grouped = df_glm_mice.groupby('binned_ev').agg(
    ev_mean=('evidence', 'mean'),
    p_right_mean=(data_label, 'mean')
    ).dropna()
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    #print(ev_means)
    #print(p_right_mean)
    ax.plot(ev_means,psychometric(ev_means), label = 'GLM Model', color = 'grey')
    ax.plot(ev_means, p_right_mean, marker = 'o', label = 'Data', color = 'black')



def psychometric_data(ax,df_glm_mice, GLM_df,regressors_string,data_label):
    #we will first compute the evidence:
    regressors_vect = regressors_string.split(' + ')
    coefficients = GLM_df['coefficient']
    df_glm_mice['evidence'] = coefficients['Intercept']
    for j in range(len(regressors_vect)):
        df_glm_mice['evidence']+= coefficients[regressors_vect[j]]*df_glm_mice[regressors_vect[j]]
    #psychometric_fit(ax,df_glm_mice)
    psychometric_plot(ax,df_glm_mice,data_label)
    

def GLM_regressors_prob_r(df,n_regressors):
    # Prepare df columns
    # Converting the 'outcome' column to boolean values
    select_columns = ['reward', 'actions', 'iti','prob_r', 'split_label']
    df_glm = df.loc[:, select_columns].copy()
    # subtract 2 from actions to get 0 for left and 1 for right
    df_glm['outcome_bool'] = df_glm['reward']
    df_glm['choice'] = df_glm['actions']-2
    #keep only the right-left actions
    df_glm = df_glm[df_glm['choice'] >= 0]

    # calculate correct_choice regressor r_+
    df_glm.loc[df_glm['outcome_bool'] == 0, 'r_plus']  = 0
    df_glm.loc[(df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 0), 'r_plus'] = -1
    df_glm.loc[(df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 1), 'r_plus'] = 1
    df_glm['r_plus'] = pd.to_numeric(df_glm['r_plus'], errors='coerce')

    # same as above but for r_-
    # define conditions
    df_glm.loc[df_glm['outcome_bool'] == 1, 'r_minus']  = 0
    df_glm.loc[(df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 0), 'r_minus'] = -1
    df_glm.loc[(df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 1), 'r_minus'] = 1
    df_glm['r_minus'] = pd.to_numeric(df_glm['r_minus'], errors='coerce')

    # Creating columns for previous trial results
    regr_plus = ''
    regr_minus = ''
    for i in range(1, n_regressors):
        df_glm[f'r_plus_{i}'] = df_glm['r_plus'].shift(i)
        df_glm[f'r_minus_{i}'] = df_glm['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
    regressors = regr_plus + regr_minus[:-3]

    return df_glm, regressors


def plot_GLM_prob_r(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df['regressor'].str.contains('r_plus'), 'coefficient']
    r_minus = GLM_df.loc[GLM_df['regressor'].str.contains('r_minus'), 'coefficient']
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='.', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='.', color='teal', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='r+'),
        mpatches.Patch(color='teal', label='r-')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')

def glm_prob_r_analysis(df,split,n_regressors):        
    # Prepare data for GLM analysis
    # Get all training data across all splits
    df_glm, regressors_string = GLM_regressors_prob_r(df,n_regressors)
    df_train = df_glm[df_glm['split_label'] == f'train_{split+1}']
    df_test = df_glm[df_glm['split_label'] == f'test_{split+1}']
    regressor_list = [x.strip() for x in regressors_string.split(' + ')] + ['choice']
    # Create subset DataFrame with only these regressors
    df_vif = df_glm[regressor_list].copy()
    print(calculate_vif(df_vif))
    mM_logit = smf.logit(formula='choice ~ ' + regressors_string,data=df_train).fit()

    
    # Create results dataframe
    GLM_df = pd.DataFrame({
        'coefficient': mM_logit.params,
        'std_err': mM_logit.bse,
        'z_value': mM_logit.tvalues,
        'p_value': mM_logit.pvalues,
        'conf_Interval_Low': mM_logit.conf_int()[0],
        'conf_Interval_High': mM_logit.conf_int()[1]
    })
    df_test['pred_prob'] = mM_logit.predict(df_test)
    #Create a DataFrame with the avaluation metrics
    y_true = df_test['choice'][n_regressors:]   # True binary outcomes
    y_pred_prob = mM_logit.predict(df_test)[n_regressors:]  # Predicted probabilities (change this tot the test set)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    np.random.seed(42) 
    y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int) # We may use the multinomial here to choose with probability (sampling)

    metrics_dict = {
        # Log-likelihood
        "log_likelihood": mM_logit.llf,
        "log_likelihood_per_obs": mM_logit.llf / len(y_true),
        
        # Information criteria
        "AIC": mM_logit.aic,
        "BIC": mM_logit.bic,
        
        # Pseudo R-squared
        "pseudo_r2_mcfadden": mM_logit.prsquared,  # McFadden's pseudo R²
        "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),  # Cox-Snell
        "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                            (1 - np.exp(2 * mM_logit.llnull / len(y_true))),  # Nagelkerke
        
        # Classification metrics (threshold=0.5)
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class),
        "recall": recall_score(y_true, y_pred_class),
        "f1_score": f1_score(y_true, y_pred_class),
        "accuracy_bis": accuracy_score(y_true, y_pred_class_mult),
        "precision_bis": precision_score(y_true, y_pred_class_mult),
        "recall_bis": recall_score(y_true, y_pred_class_mult),
        "f1_score_bis": f1_score(y_true, y_pred_class_mult),
        
        # Probability-based metrics
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "brier_score": brier_score_loss(y_true, y_pred_prob),
    }
    GLM_metrics = pd.DataFrame([metrics_dict]) 
    return GLM_df,regressors_string,df_glm, GLM_metrics

        

def GLM_regressors_switch(df,n_regressors):
    """
    Summary:
    This function processes the data needed to obtain the regressors and derives the 
    formula for the glm

    Args:
        df ([Dataframe]): [dataframe with experimental data]
        n ([int]): [number of trials back considered]

    Returns:
        new_df([Dataframe]): [dataframe with processed data restrcted to the regression]
        regressors_string([string]) :  [regressioon formula]
    """
    select_columns = ['reward', 'actions', 'iti', 'prob_r','split_label']
    df_glm = df.loc[:, select_columns].copy()
    # subtract 2 from actions to get 0 for left and 1 for right
    df_glm['outcome_bool'] = df_glm['reward']
    df_glm['choice'] = df_glm['actions']-2
    #keep only the right-left actions
    df_glm = df_glm[df_glm['choice'] >= 0]
    # Select the columns needed for the regressors
    df_glm = df_glm.copy()

    #TODO: A column indicating significant columns will be constructed
    # session_counts = df_glm['session'].value_counts()
    # mask = df_glm['session'].isin(session_counts[session_counts > 50].index)
    # df_glm['sign_session'] = 0
    # df_glm.loc[mask, 'sign_session'] = 1
    # df_glm = df_glm[df_glm['sign_session'] == 1]

    #prepare the switch regressor
    df_glm['choice_1'] = df_glm['choice'].shift(1)
    df_glm.loc[(df_glm['choice'] == df_glm['choice_1']), 'switch_num'] = 0
    df_glm.loc[(df_glm['choice'] != df_glm['choice_1']), 'switch_num'] = 1


    # Last trial reward
    df_glm['last_trial'] = df_glm['outcome_bool'].shift(1)
    # build the regressors for previous trials
    rss_plus = ''
    rss_minus = ''
    rds_plus = ''
    for i in range(2, n_regressors + 1):
        df_glm[f'choice_{i}'] = df_glm['choice'].shift(i)
        df_glm[f'outcome_bool_{i}'] = df_glm['outcome_bool'].shift(i)
        
        #prepare the data for the error_switch regressor rss_-
        df_glm.loc[(df_glm[f'choice_{i}'] == df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 0), f'rss_minus{i}'] = 1
        df_glm.loc[(df_glm[f'choice_{i}'] == df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 1), f'rss_minus{i}'] = 0
        df_glm.loc[df_glm[f'choice_{i}'] != df_glm['choice_1'], f'rss_minus{i}'] = 0
        df_glm[f'rss_minus{i}'] = pd.to_numeric(df_glm[f'rss_minus{i}'], errors='coerce')
        #prepare the data for the error_switch regressor rss_-
        df_glm.loc[(df_glm[f'choice_{i}'] == df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 1), f'rss_plus{i}'] = 1
        df_glm.loc[(df_glm[f'choice_{i}'] == df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 0), f'rss_plus{i}'] = 0
        df_glm.loc[df_glm[f'choice_{i}'] != df_glm['choice_1'], f'rss_plus{i}'] = 0
        df_glm[f'rss_plus{i}'] = pd.to_numeric(df_glm[f'rss_plus{i}'], errors='coerce')
        rss_plus += f'rss_plus{i} + '
        rss_minus += f'rss_minus{i} + '

        #prepare the data for the error_switch regressor rds_+
        df_glm.loc[(df_glm[f'choice_{i}'] != df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 1), f'rds_plus{i}'] = 1
        df_glm.loc[(df_glm[f'choice_{i}'] != df_glm['choice_1']) & (df_glm[f'outcome_bool_{i}'] == 0), f'rds_plus{i}'] = 0
        df_glm.loc[df_glm[f'choice_{i}'] == df_glm['choice_1'], f'rds_plus{i}'] = 0
        df_glm[f'rss_plus{i}'] = pd.to_numeric(df_glm[f'rss_plus{i}'], errors='coerce')
        rds_plus += f'rds_plus{i} + '
    regressors_string = rss_plus + rss_minus + rds_plus + 'last_trial'
    df_glm = df_glm.copy()

    return df_glm, regressors_string

def plot_GLM_prob_switch(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    rss_plus = GLM_df.loc[GLM_df['regressor'].str.contains('rss_plus'), 'coefficient']
    if GLM_df['regressor'].str.contains('rss_plus10').any():
        rss_plus = pd.Series(np.roll(rss_plus, -1)) 
    rss_minus = GLM_df.loc[GLM_df['regressor'].str.contains('rss_minus'), 'coefficient']
    if GLM_df['regressor'].str.contains('rss_minus10').any():
        rss_minus = pd.Series(np.roll(rss_minus, -1)) 
    rds_plus = GLM_df.loc[GLM_df['regressor'].str.contains('rds_plus'), 'coefficient']
    if GLM_df['regressor'].str.contains('rds_plus10').any():
        rds_plus = pd.Series(np.roll(rds_plus, -1)) 
    last_trial = GLM_df.loc[GLM_df['regressor'].str.contains('last_trial'), 'coefficient']
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(rss_plus)], rss_plus, marker='o', color='indianred', alpha=alpha)
    ax.plot(orders[:len(rss_minus)], rss_minus, marker='o', color='teal', alpha=alpha)
    ax.plot(orders[:len(rds_plus)], rds_plus, marker='o', color='red', alpha=alpha)
    ax.plot(orders[:len(last_trial)], last_trial, marker='o', color='green', alpha=alpha)

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='rss+'),
        mpatches.Patch(color='teal', label='rss-'),
        mpatches.Patch(color='red', label='rds+'),
        mpatches.Patch(color='green', label='last_trial')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')
def psychometric_plot(ax,df_glm_mice, data_label):
    n_bins = 10
    #equiespaced bins
    bins = np.linspace(df_glm_mice['evidence'].min(), df_glm_mice['evidence'].max(), n_bins)
    df_glm_mice['binned_ev'] = pd.cut(df_glm_mice['evidence'], bins=bins)
    #equipopulated bins
    #df_glm_mice['binned_ev'] = pd.qcut(df_glm_mice['evidence'], n_bins,duplicates='drop')
    #bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
    #print histograms
    histogram = 0
    if histogram:
        bin_counts = df_glm_mice['binned_ev'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        bin_counts.plot(kind='bar', width=0.8, color='skyblue', edgecolor='black')
        plt.title('Histogram of Elements in Each Bin', fontsize=16)
        plt.xlabel('Bin Interval', fontsize=14)
        plt.ylabel('Number of Elements', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    grouped = df_glm_mice.groupby('binned_ev').agg(
    ev_mean=('evidence', 'mean'),
    p_right_mean=(data_label, 'mean')
    ).dropna()
    ev_means = grouped['ev_mean'].values
    p_right_mean = grouped['p_right_mean'].values
    #print(ev_means)
    #print(p_right_mean)
    ax.plot(ev_means,psychometric(ev_means), label = 'GLM Model', color = 'grey')
    ax.plot(ev_means, p_right_mean, marker = 'o', label = 'Data', color = 'black')



def psychometric_data(ax,df_glm_mice, GLM_df,regressors_string,data_label):
    #we will first compute the evidence:
    regressors_vect = regressors_string.split(' + ')
    GLM_df = GLM_df.copy()
    GLM_df = GLM_df.reset_index(drop=True)
    coefficients = GLM_df['coefficient']
    df_glm_mice['evidence'] = GLM_df.loc[GLM_df['regressor'] == 'Intercept', 'coefficient'].values[0]
    
    for j in range(len(regressors_vect)):
        coef = GLM_df.loc[GLM_df['regressor'] == regressors_vect[j], 'coefficient']
        df_glm_mice['evidence']+= coef.values[0]*df_glm_mice[regressors_vect[j]]
    psychometric_plot(ax,df_glm_mice,data_label)

def glm_switch_analysis(df,split,n_regressors):
    df_glm, regressors_string = GLM_regressors_switch(df,n_regressors)
    df_train = df_glm[df_glm['split_label'] == f'train_{split+1}']
    df_test = df_glm[df_glm['split_label'] == f'test_{split+1}']
    regressor_list = [x.strip() for x in regressors_string.split(' + ')] + ['switch_num']
    # Create subset DataFrame with only these regressors
    df_vif = df_glm[regressor_list].copy()
    print(calculate_vif(df_vif))

    #TODO: Implement cross-validation (create a train and test set)
    mM_logit = smf.logit(formula='switch_num ~ ' + regressors_string,data=df_train).fit()    
    # Create results dataframe
    GLM_df = pd.DataFrame({
        'coefficient': mM_logit.params,
        'std_err': mM_logit.bse,
        'z_value': mM_logit.tvalues,
        'p_value': mM_logit.pvalues,
        'conf_Interval_Low': mM_logit.conf_int()[0],
        'conf_Interval_High': mM_logit.conf_int()[1]
    })
    df_test['pred_prob'] = mM_logit.predict(df_test)
    #Create a DataFrame with the avaluation metrics
    y_true = df_test['switch_num'][n_regressors:]   # True binary outcomes
    y_pred_prob = mM_logit.predict(df_test)[n_regressors:]  # Predicted probabilities (change this tot the test set)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    np.random.seed(42) 
    y_pred_class_mult = (np.random.rand(len(y_pred_prob)) < y_pred_prob).astype(int) # We may use the multinomial here to choose with probability (sampling)
    #compute BIC from the log-likelihood
    # Compute BIC from the log-likelihood
    # BIC = -2 * log-likelihood + k * log(n)
    metrics_dict = {
        # Log-likelihood
        "log_likelihood": mM_logit.llf,
        "log_likelihood_per_obs": mM_logit.llf / len(y_true),   
        
        # Information criteria
        "AIC": mM_logit.aic,
        "BIC": mM_logit.bic,
        
        # Pseudo R-squared
        "pseudo_r2_mcfadden": mM_logit.prsquared,  # McFadden's pseudo R²
        "pseudo_r2_cox_snell": 1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true)),  # Cox-Snell
        "pseudo_r2_nagelkerke": (1 - np.exp(-2 * (mM_logit.llf - mM_logit.llnull) / len(y_true))) / 
                            (1 - np.exp(2 * mM_logit.llnull / len(y_true))),  # Nagelkerke
        
        # Classification metrics (threshold=0.5)
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class),
        "recall": recall_score(y_true, y_pred_class),
        "f1_score": f1_score(y_true, y_pred_class),
        "accuracy_bis": accuracy_score(y_true, y_pred_class_mult),
        "precision_bis": precision_score(y_true, y_pred_class_mult),
        "recall_bis": recall_score(y_true, y_pred_class_mult),
        "f1_score_bis": f1_score(y_true, y_pred_class_mult),
        
        # Probability-based metrics
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "brier_score": brier_score_loss(y_true, y_pred_prob),
    }
    GLM_metrics = pd.DataFrame([metrics_dict]) 
    return GLM_df,regressors_string,df_glm, GLM_metrics