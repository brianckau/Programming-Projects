from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.pipeline import *
from sklearn.preprocessing import *

import pandas as pd
import numpy as np

def lasso_regression_model_for_prediction(combined_df, n_splits, return_xmonthlater=0):

    if return_xmonthlater > 0:
        Y = combined_df.iloc[:,0].shift(-return_xmonthlater).iloc[:-return_xmonthlater]
        X = combined_df.iloc[:-return_xmonthlater,1:]
    elif return_xmonthlater == 0:
        Y = combined_df.iloc[:,0]
        X = combined_df.iloc[:,1:]
    else:
        print("return_xmonthlater must be >= 0.")

    IC = make_scorer(information_coefficient,greater_is_better=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('lasso', Lasso())])
    
    param_grid = {'lasso__alpha':[0.001,0.01,0.1,1.0,10,100]}
    
    optimal_model = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=tscv,
                               scoring=IC,
                               n_jobs=-1,
                               verbose=1,
                               refit=True)
    
    optimal_model.fit(X,Y)

    print('------------------------------------------------------')
    print(f'The IC of the model is: {round(optimal_model.best_score_,5)}')
    if np.abs(optimal_model.best_score_) >= 0.08:
        print(f'the model is extremely useful')
    elif np.abs(optimal_model.best_score_) > 0.05:
        print(f'the model is highly useful')
    elif np.abs(optimal_model.best_score_) > 0.02:
        print(f'the model is good')
    elif np.abs(optimal_model.best_score_) > 0:
        print(f'the model is weak')
    else:
        print(f'the model captures only random noise')

    return optimal_model



def information_coefficient(y_true, y_pred):
    if np.std(y_pred) == 0:
        return 0.0
    else:
        return np.corrcoef(y_true, y_pred)[0,1]