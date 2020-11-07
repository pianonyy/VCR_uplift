import pandas as pd
import numpy as np
def error(group_type, X, Y):
    Y=np.array(Y)
    
    X['uplift_score'] = (X['uplift_score'] > 0).astype(int)
    
    if group_type == 'treat':
        X['error'] = (X['uplift_score'] != Y).astype(int)
    if group_type == 'control':
        X['error'] = (X['uplift_score'] == Y).astype(int)

    return X['error']
""" _____________________________________ADABOOST IMPLEMENTATION____________________________________________"""

def uplift_adaboost(n_estimators, estimator, X_train, Y_train, treatment_train):
    
    train_size = X_train.shape[0]
    

    #  check_consistent_length(X_train, Y_train, treatment_train)
    #  check_consistent_length(X_test, Y_test, treatment_test)

    treat_records_x = X_train[treatment_train == 1] 
    control_records_x = X_train[treatment_train == 0]
    treat_records_y = Y_train[treatment_train == 1] 
    control_records_y = Y_train[treatment_train == 0]
    
    # 1 intialize weights
    treat_weights = np.ones(shape = treat_records_x.shape[0])
    control_weights = np.ones(shape = control_records_x.shape[0])

    # 2 for
    weight_array_list_treatment =[]
    weight_array_list_control = []
    
    total_betas = []

    #weight_array_list_control.append(control_weights)
    #weight_array_list_treatment.append(treatment_weights)
    
    for i in range(0,n_estimators):
        
        treat_weights = treat_weights / (treat_weights.sum() + control_weights.sum())
        control_weights = control_weights / (treat_weights.sum() + control_weights.sum())
        #add weights to X_train
        
        treat_records_x['weights'] = treat_weights
        control_records_x['weights'] = control_weights
        
        #print(treat_records_x)
        X_train = pd.concat([treat_records_x, control_records_x])
        Y_train = np.concatenate((treat_records_y, control_records_y), axis=None)
        
        estimator.fit(X_train, treatment_train, Y_train)
        
        uplift_score_t = estimator.predict_uplift(treat_records_x)
        uplift_score_c = estimator.predict_uplift(control_records_x)

        treat_records_x['uplift_score'] = uplift_score_t
        control_records_x['uplift_score'] = uplift_score_c
        
        error_T = error(group_type = 'treat',X = treat_records_x, Y = treat_records_y)
        error_C = error(group_type ='control', X = control_records_x,Y =control_records_x)
        

        #compute treat and control errors (5)
        treat_error = treat_records_x.loc[error_T == 1,'weights'].sum() / treat_records_x['weights'].sum()
        control_error = control_records_x.loc[error_C == 1,'weights'].sum() / control_records_x['weights'].sum()
        
        #compute relative sizes (1)
        p_t = treat_records_x['weights'].sum() / (treat_records_x['weights'].sum() + control_records_x['weights'].sum())
        p_c = control_records_x['weights'].sum() / (treat_records_x['weights'].sum() + control_records_x['weights'].sum())
        
        #total error (5)
        total_error = p_t * treat_error + p_c * control_error
        
        #compute betas (d)
        beta = total_error / (1 - total_error)
        if (beta == 1) or (treat_error <=0 or treat_error >= 0.5) or (treat_error <= 0 or treat_error >= 0.5):
            treat_weights = np.random.normal(loc=0.5, scale=0.5,size = treat_records_x.shape[0])
            control_weights = np.random.normal(loc=0.5, scale=0.5,size = control_records_x.shape[0])
            continue
            
        #update weights (f) (g)
        
        treat_weights = treat_weights * (beta ** ((treat_records_x['uplift_score'] == treat_records_y).astype(int)))
        control_weights = control_weights * (beta ** ((control_records_x['uplift_score'] == (1 - control_records_y)).astype(int)))
        
        total_betas.append(beta)
    print(total_betas)