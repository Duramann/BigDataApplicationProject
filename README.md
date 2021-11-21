Application of Big Data Project - NOUAR Alexandre & DURA Théo 
==============================

This is the project for the course named "Application of big data"

Project Organization
------------
├── README.md     
├── data              
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- The Sphinx Documentation.
│
├── models             <- Contains saved model as .pkl
│
├── mlruns             <- Contains mlflow runs (Our experiment and runs are on the "1" folder) 
│
├── notebooks          <- Jupyter notebooks
│
├── predictions        <- Contains the predictions of the three models as csv files   
│
├── conda.yml          <- The conda environnement properties                      
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │   
    ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   └── build_features.py
    │
    └── models         <- Scripts to train models and then use trained models to make
        │                 predictions
        │
        │
        ├── predict_model.py
        ├── train_model.py
        └── train_mlflow.py
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
