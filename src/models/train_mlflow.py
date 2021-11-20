import sys
import os


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost
import mlflow.sklearn

def main():
       
    ## Allow us to parse value for the model when launching the script    
    eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    colsample = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    subsample = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    # Load data
    try:
        df_train = pd.read_csv('data/processed/train_data.csv')
    except (FileNotFoundError, IOError):
            print("Unable to fetch data, please run this script from BigDataApplicationProject folder.")
            sys.exit()
    
    ## Separate Target from freatures
    X = df_train.drop(columns = ['TARGET'])
    Y = df_train['TARGET']
    
    
    ## Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # enable auto logging
    mlflow.xgboost.autolog()
    
    # Set mlflow experiment name
    mlflow.set_experiment("XGBOOST")
    
    ## Start mlflow run
    with mlflow.start_run():
        
        ## Define the model
        XGB = XGBClassifier(objective='binary:logistic', eval_metric="logloss", use_label_encoder=False, eta=eta, subsample=subsample
                           , colsample_bytree=colsample)
        ## Train the model
        XGB.fit(X_train, y_train)

        # Make prediction and then evaluate the model with accuracy, precision and confusion matrix
        y_pred = XGB.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test,y_pred)
        prec = precision_score(y_test, y_pred)
               
        ## Make confusion matrix into a png to store it    
        hm = sb.heatmap(cm,annot=True, fmt='g')
        plt.savefig('hm.png')
        mlflow.log_artifact("hm.png")
        os.remove('hm.png')
        
        ## log metrics and choosen parameters
        mlflow.log_metrics({"accuracy": acc,"precision": prec})
        mlflow.log_param("eta", eta)
        mlflow.log_param("colsample_bytree", colsample)
        mlflow.log_param("subsample", subsample)
        
        ## Store the model
        mlflow.sklearn.log_model(XGB, "XGB_model")

if __name__ == "__main__":
    main()