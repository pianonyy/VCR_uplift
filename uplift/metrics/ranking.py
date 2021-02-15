import warnings
import numpy as np
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import auc
import pandas as pd

import plotly.express as px

import scipy.stats as stats

from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)


#percentile approach to qini measurement

def declare_tc(df):
   
    
    df['target_class'] = 0 
    
    df.loc[(df.treatment == 0) & (df.target != 0),'target_class'] = 1 
  
    df.loc[(df.treatment != 0) & (df.target == 0),'target_class'] = 2 
    
    df.loc[(df.treatment != 0) & (df.target != 0),'target_class'] = 3 
    return df


def qini_rank(Y_test_visit, pred, treatment_test):
    
    uplift = pd.DataFrame({'target':Y_test_visit,'treatment':treatment_test, 'uplift_score':pred})
    
    uplift = declare_tc(uplift)
    
    ranked = pd.DataFrame({'n':[], 'target_class':[]})
    
    ranked['target_class'] = uplift['target_class']
    ranked['uplift_score'] = uplift['uplift_score']
    
    
  
    ranked['n'] = ranked.uplift_score.rank(pct=True, ascending=False)
   
   
    ranked = ranked.sort_values(by='n').reset_index(drop=True)

    
    print(ranked)
    return ranked


def qini_eval(ranked):
    
    
    uplift_model, random_model = ranked.copy(), ranked.copy()
    # Using Treatment and Control Group to calculate the uplift 
    C, T = sum(ranked['target_class'] <= 1), sum(ranked['target_class'] >= 2)
    ranked['cr'] = 0
    ranked['tr'] = 0
    ranked.loc[ranked.target_class == 1,'cr'] = 1
    ranked.loc[ranked.target_class == 3,'tr'] = 1
    ranked['cr/c'] = ranked.cr.cumsum() / C
    ranked['tr/t'] = ranked.tr.cumsum() / T
    # Calculate and put the uplift and random value into dataframe
    uplift_model['uplift'] = round(ranked['tr/t'] - ranked['cr/c'],5)
    random_model['uplift'] = round(ranked['n'] * uplift_model['uplift'].iloc[-1],5)
    
    
    # Add start point
    q0 = pd.DataFrame({'n':0, 'uplift':0, 'target_class': None}, index =[0])
    uplift_model = pd.concat([q0, uplift_model],sort=False).reset_index(drop = True)
    random_model = pd.concat([q0, random_model],sort=False).reset_index(drop = True)  
    # Add model name & concat
    uplift_model['model'] = 'Uplift model'
    random_model['model'] = 'Random model'
    merged = pd.concat([uplift_model, random_model]).sort_values(by='n').reset_index(drop = True)
    merged = merged.groupby(['n','model'],as_index=False).mean()
    print(merged)
    return merged


def qini_plot(merged):
    fig = px.line(merged, x="n", y="uplift", color='model', labels={
                     "n": "ratio",
                     "uplift": "uplift"
                    }
                )
    
    fig.show()
    

    


def qini_percentile(Y_test_visit,pred,treatment_test):
  
    ranked = qini_rank(Y_test_visit,pred,treatment_test)
    merged = qini_eval(ranked)
    qini_plot(merged)
   



#absolute numbers qini curve
def qini_curve(y_true, uplift, treatment): 

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]

    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    #print(threshold_indices.size)

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = y_trmnt - y_ctrl * np.divide(num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0)
    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
       
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values

def perfect_qini_curve(y_true, treatment):
  
    check_consistent_length(y_true, treatment)
    n_samples = len(y_true)

    y_true, treatment = np.array(y_true), np.array(treatment)

    
    
    x_perfect, y_perfect = qini_curve(
            y_true, y_true * treatment - y_true * (1 - treatment), treatment
    )
    

    return x_perfect, y_perfect



def qini_auc_score(y_true, uplift, treatment, negative_effect=True):
   
    check_consistent_length(y_true, uplift, treatment)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    treatment_count = np.count_nonzero(treatment == 1)


    x_model, y_model = qini_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_qini_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])
    
    # print(np.size(treatment))
    #x_baseline, y_baseline = np.array([np.arange(0, np.size(treatment))]), np.array([0, y_perfect[-1]])
    

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_model = auc(x_model, y_model) - auc_score_baseline

    return auc_score_model







def uplift_at_k(y_target, prediction_score, treatment, rate=0.3):

    check_consistent_length(y_target, prediction_score, treatment)
    prediction_score = np.array(prediction_score)
    order = np.argsort(prediction_score, kind='mergesort')[::-1]
    # order = np.argsort(-prediction_score)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = y_target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = y_target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score

def response_rate_by_percentile(y_true, uplift, treatment, group, strategy='overall', bins=10):

    check_consistent_length(y_true, uplift, treatment)
    

    group_types = ['treatment', 'control']
    strategy_methods = ['overall', 'by_group']
    
    n_samples = len(y_true)
    
    if group not in group_types:
        raise ValueError(f'Response rate supports only group types in {group_types},'
                         f' got {group}.') 

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')
    
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer. Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')
    
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    order = np.argsort(uplift, kind='mergesort')[::-1]

    trmnt_flag = 1 if group == 'treatment' else 0
    
    if strategy == 'overall':
        y_true_bin = np.array_split(y_true[order], bins)
        trmnt_bin = np.array_split(treatment[order], bins)
        
        group_size = np.array([len(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])
        response_rate = np.array([np.mean(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])

    else:  # strategy == 'by_group'
        y_bin = np.array_split(y_true[order][treatment[order] == trmnt_flag], bins)
        
        group_size = np.array([len(y) for y in y_bin])
        response_rate = np.array([np.mean(y) for y in y_bin])

    variance = np.multiply(response_rate, np.divide((1 - response_rate), group_size))

    return response_rate, variance, group_size
  

def Kendall_rank_correlation(y_true, uplift, treatment, strategy='overall', bins=10, std=False):

    check_consistent_length(y_true, uplift, treatment)
    

    strategy_methods = ['overall', 'by_group']

    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(f'Response rate supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    if not isinstance(total, bool):
        raise ValueError(f'Flag total should be bool: True or False.'
                         f' Invalid value total: {total}')

    if not isinstance(std, bool):
        raise ValueError(f'Flag std should be bool: True or False.'
                         f' Invalid value std: {std}')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer.'
                         f' Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl
    uplift_variance = variance_trmnt + variance_ctrl

    percentiles = [round(p * 100 / bins, 1) for p in range(1, bins + 1)]

    order = np.argsort(uplift, kind='mergesort')[::-1]
    uplift_by_bins = np.array_split( uplift[order], bins)

    Output = [] 
    

    for i in range(len(uplift_by_bins)): 
        Output.append(np.mean(uplift_by_bins[i])) 
    
    

    df = pd.DataFrame({
        'percentile': percentiles,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'teoretical_uplift': uplift_scores,
        'predicted_uplift': Output
    })

    tau, p_value = stats.kendalltau(df['teoretical_uplift'], df['predicted_uplift'] )

    print("Kendal uplift rank correlation = ", tau, "with p_value = ", p_value)
    
    return df