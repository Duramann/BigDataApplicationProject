import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

try:
    df_train = pd.read_csv('../../data/processed/train_data.csv')
    df_test = pd.read_csv('../../data/processed/test_data.csv')
except (FileNotFoundError, IOError):
            print("Unable to fetch data, please run this script from BigDataApplicationProject/src/models folder.")
            sys.exit()
    

X = df_train.drop(columns = ['TARGET', 'Unnamed: 0'])
Y = df_train['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, Y)

## XGBOOST MODEL :

from xgboost import XGBClassifier

XGB = XGBClassifier(objective = "binary:logistic", use_label_encoder=False)
XGB.fit(X_train, y_train)

filenameXGB = '../../models/XGB_model.sav'
pickle.dump(XGB, open(filenameXGB, 'wb')) 

## RANDOM FOREST :

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=200)
RFC.fit(X_train, y_train)

filenameRFC = '../../models/RFC_model.sav'
pickle.dump(RFC, open(filenameRFC, 'wb')) 

## GRADIENT BOOSTING MODEL :

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)

filenameGBC = '../../models/GBC_model.sav'
pickle.dump(GBC, open(filenameGBC, 'wb')) 