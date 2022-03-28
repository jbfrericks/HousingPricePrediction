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

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
import xgboost as xgb
import xgbooster
import RandomForest
import os

import gdbt

sb.set_style('whitegrid') # plot style
#plt.rcParams['figure.figsize'] = (20, 10) # plot size

# TRAINING DATA
df = pd.read_csv('HousingData.csv')
df.fillna(value = 0, inplace = True)
dataTrain = df._get_numeric_data()
print(dataTrain.head(10))
XTrain, YTrain = dataTrain.iloc[:,:-1], dataTrain.iloc[:,-1]

""" #TESTING DATA
df1 = pd.read_csv('test.csv')
df1.set_index('Id', inplace = True)

#df1['new']=df1.MSZoning.apply(lambda x: np.where(x.isdigit(),x,'0'))

dataTest = df1._get_numeric_data()
print(dataTest.head(10))
XTest, YTest = dataTest.iloc[:,:-1], dataTest.iloc[:,-1] """

#SPLITS DATA INTO TESTING AND TRAINING SETS
X_train, X_test, y_train, y_test = train_test_split(XTrain, YTrain, test_size=0.2, random_state=123) 

#XGBOOST REGRESSOR WITH RMSE
#xgbooster.xbgooster(X_train, y_train, X_test, y_test)

#RANDOM FOREST
#RandomForest.randForest(X_train, y_train, X_test, y_test)

#GDBT
gdbt.gdbt(X_train, y_train, X_test, y_test)
