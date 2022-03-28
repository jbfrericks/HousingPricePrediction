#John Frericks
#University of Georgia
#CSCI 8265



from sklearn.ensemble import GradientBoostingRegressor
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
import xgboost as xgb
import xgbooster
import RandomForest
import os


def gdbt(dataTrainX, dataTrainY, dataTestX, dataTestY):
    regressor = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=3,
        learning_rate=1.0
    )
    regressor.fit(dataTrainX, dataTrainY)

    errors = [mean_squared_error(dataTestY, y_pred) for y_pred in regressor.staged_predict(dataTestX)]
    best_n_estimators = np.argmin(errors)

    best_regressor = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=best_n_estimators,
        learning_rate=1.0
    )
    best_regressor.fit(dataTrainX, dataTrainY)

    y_pred = best_regressor.predict(dataTestX)
    print("mean abs. error: ", (mean_absolute_error(dataTestY, y_pred)))