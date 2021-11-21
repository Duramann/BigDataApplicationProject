import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

##Load data
try:
    df_train = pd.read_csv('data/processed/train_data.csv')
    df_test = pd.read_csv('data/processed/test_data.csv')
except (FileNotFoundError, IOError):
            print("Unable to fetch data, please run this script from BigDataApplicationProject folder.")
            sys.exit()
    
## Adjust data format :
X = df_train.drop(columns = ['TARGET'])
Y = df_train['TARGET']

## Train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y)

## XGBOOST MODEL :

from xgboost import XGBClassifier

## Building :
XGB = XGBClassifier(objective='binary:logistic', eval_metric="logloss", use_label_encoder=False, eta=0.3, subsample=1
                           , colsample_bytree=1)
## Training :
XGB.fit(X_train, y_train)

## Model storing 
filenameXGB = 'models/base_XGB_model.pkl'
pickle.dump(XGB, open(filenameXGB, 'wb')) 

## RANDOM FOREST :

from sklearn.ensemble import RandomForestClassifier

## Building
RFC = RandomForestClassifier()

## Training
RFC.fit(X_train, y_train)

## Model storing
filenameRFC = 'models/RFC_model.pkl'
pickle.dump(RFC, open(filenameRFC, 'wb')) 

## GRADIENT BOOSTING MODEL :

from sklearn.ensemble import GradientBoostingClassifier

## Building
GBC = GradientBoostingClassifier()

## Training
GBC.fit(X_train, y_train)

## Model storing
filenameGBC = 'models/GBC_model.pkl'
pickle.dump(GBC, open(filenameGBC, 'wb')) 