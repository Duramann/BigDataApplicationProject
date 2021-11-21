Project organization
--------------------

The project is organized as follows:


├── README.md     |br|
├── data          |br|   
│      ├── interim        <- Intermediate data that has been transformed. |br|
│      ├── processed      <- The final, canonical data sets for modeling. |br|
│      └── raw            <- The original, immutable data dump. |br|
│ |br|
├── docs               <- The Sphinx Documentation. |br|
│ |br|
├── models             <- Contains saved model as .pkl |br|
│ |br|
├── mlruns             <- Contains mlflow runs (Our experiment and runs are on the "1" folder)  |br|
│ |br|
├── notebooks          <- Jupyter notebooks |br|
│ |br|
├── predictions        <- Contains the predictions of the three models as csv files |br|  
│ |br|
├── conda.yml          <- The conda environnement properties  |br|                   
│ |br|
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported |br|
└── src                <- Source code for use in this project. |br|
       ├── __init__.py    <- Makes src a Python module |br|
       │   |br|
       ├── features       <- Scripts to turn raw data into features for modeling |br|
       │   │ |br|
       │   └── build_features.py |br|
       │ |br|
       └── models         <- Scripts to train models and then use trained models to make predictions |br|
           │ |br|
           ├── predict_model.py |br|
           ├── train_model.py |br|
           └── train_mlflow.py |br|
 
 

 .. |br| raw:: html

      <br>