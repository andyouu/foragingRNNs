import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def GLM_regressors(df):
    # Prepare df columns
    # Converting the 'outcome' column to boolean values
    select_columns = ['reward', 'actions', 'iti']
    df_glm = df.loc[:, select_columns].copy()
    # subtract 2 from actions to get 0 for left and 1 for right
    df_glm['outcome_bool'] = df_glm['reward']
    df_glm['choice'] = df_glm['actions']-2
    df_glm['actions'][df_glm['actions']<0] = np.nan

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
    max_shift = 5
    regr_plus = ''
    regr_minus = ''
    for i in range(1, max_shift):
        df_glm[f'r_plus_{i}'] = df_glm['r_plus'].shift(i)
        df_glm[f'r_minus_{i}'] = df_glm['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
    regressors = regr_plus + regr_minus[:-3]
    return df_glm, regressors


def plot_GLM(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.startswith('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.startswith('r_minus'), "coefficient"]
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