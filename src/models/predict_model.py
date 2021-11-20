import sys

import pandas as pd
import mlflow
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import mlflow.xgboost
import xgboost

## Load data

try:
    df_test = pd.read_csv('data/processed/test_data.csv')
except (FileNotFoundError, IOError):
            print("Unable to fetch data, please run this script from BigDataApplicationProject folder.")
            sys.exit();
            
x_test = df_test

## Load gbc and rfc model
try:
    gbc = pickle.load(open('models/GBC_model.pkl', 'rb'))
    rfc = pickle.load(open('models/RFC_model.pkl', 'rb'))
except (FileNotFoundError, IOError):
            print("Unable to fetch data, please train your models first by running src/models/train_model.py first.")
            sys.exit();

## Load xgb model by choosing the best one from the mlflow experiments :
experiment_name = "XGBOOST"
current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id=current_experiment['experiment_id']

df = mlflow.search_runs([experiment_id], order_by=["metrics.precision DESC"])
best_run_id = df.loc[0,'run_id']

xgb = pickle.load(open('mlruns/1/'+best_run_id+'/artifacts/XGB_model/model.pkl', 'rb'))

## Save best model in models folder

filenameXGB = 'models/best_XGB_model.pkl'
pickle.dump(xgb, open(filenameXGB, 'wb')) 

## Do the predictions :

y_predict_GBC = gbc.predict(x_test)
y_predict_GBC = pd.DataFrame(y_predict_GBC, columns=['Predicted value'])

y_predict_RFC = rfc.predict(x_test)
y_predict_RFC = pd.DataFrame(y_predict_RFC, columns=['Predicted value'])

y_predict_XGB = xgb.predict(x_test)
y_predict_XGB = pd.DataFrame(y_predict_XGB, columns=['Predicted value'])

## Save the predictions into csv files :

y_predict_GBC.to_csv('predictions/GBC_predictions.csv', index=False)
y_predict_RFC.to_csv('predictions/RFC_predictions.csv', index=False)
y_predict_XGB.to_csv('predictions/XGB_predictions.csv', index=False)





