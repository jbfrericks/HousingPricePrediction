#John Frericks
#University of Georgia
#CSCI 8265


# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier


def xbgooster(dataTrainX, dataTrainY, dataTestX, dataTestY):
    
    dataTrain_dmatrix = xgb.DMatrix(data=dataTrainX,label=dataTrainY)
    dataTest_dmatrix = xgb.DMatrix(data=dataTestX,label=dataTestY)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

    
    model.fit(dataTrainX, dataTrainY)
    print(model)
    y_prediction = model.predict(dataTestX)
    print(y_prediction)
    rmse = np.sqrt(mean_squared_error(dataTestY, y_prediction))
    print("RMSE: %f" % (rmse))

    r2val = r2(dataTestY, y_prediction)
    print('R2: %f ' % (r2val))

    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=dataTrain_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    cv_results.head()

    xg_reg = xgb.train(params=params, dtrain=dataTrain_dmatrix, num_boost_round=10)

    xgb.plot_tree(xg_reg,num_trees=0)
    #plt.rcParams['figure.figsize'] = [1, 10]
    plt.show()

    xgb.plot_importance(xg_reg)
    #plt.rcParams['figure.figsize'] = [10, 10]
    plt.show()

