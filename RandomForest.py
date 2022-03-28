#John Frericks
#University of Georgia
#CSCI 8265




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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def randForest(dataTrainX, dataTrainY, dataTestX, dataTestY):
    #Create a Gaussian Classifier
    clf=RandomForestRegressor(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(dataTrainX,dataTrainY)

    print(clf.score(dataTrainX, dataTrainY))
    y_prediction=clf.predict(dataTestX)
    #print("Accuracy:",accuracy_score(dataTestY, y_prediction))
    r2val = r2(dataTestY, y_prediction)
    print((r2val))

