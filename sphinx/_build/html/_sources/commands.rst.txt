Commands
========

In this page we'll explain how to run the different script of our project.

.. _data-process:

Process the data
----------------

To process the data go into the **/BigDataApplicationProject** folder and run :

*python src/features/build_features.py*

.. _models-execution:

Models
------

Basic model training
~~~~~~~~~~~~~~~~~~~~

To train the basic models go into the **/BigDataApplicationProject** folder and run :

*python src/models/train_model.py*


Training with MLFlow
~~~~~~~~~~~~~~~~~~~~

To train the XGBoost model go into the **/BigDataApplicationProject** folder and run :

*python src/models/train_mlflow.py* *eta value* *colsample value* *subsample value* 

MLFlow ui
---------

To acces the MLFlow ui, go into the **/BigDataApplicationProject** folder and run :

*mlflow ui*

Then copy and paste the local url into a web browser to acces the ui.

Model predictions
-----------------

To predict values using our models go into the **/BigDataApplicationProject** folder and run :

*python src/models/predict_model.py*

