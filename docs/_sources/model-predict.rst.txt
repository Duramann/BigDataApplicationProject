Predictions using our models
============================

After training our base models and using MLFlow to find the best possible XGBoost model, we can then try to do predictions using our processed test dataset.

The script used to perform this task can be found at :

**src/models/predict_model.py**

For the base Random Forest and Grandient Boosting models, we just have to load our saved .pkl models using pickle.

However for the XGBoost model, since we have trained many models with MLFlow, we'll find the model that have the best precision using this code :

.. code-block:: python
	
	experiment_name = "XGBOOST"
	current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
	experiment_id=current_experiment['experiment_id']

	df = mlflow.search_runs([experiment_id], order_by=["metrics.precision DESC"])
	best_run_id = df.loc[0,'run_id']

	xgb = pickle.load(open('mlruns/1/'+best_run_id+'/artifacts/XGB_model/model.pkl', 'rb'))

We retrieve all the model with the experiment name "XGBOOST", fetch the experiment id then order all the experiments by precision descending and get its id.

We then fetch the folder containing this model and import the .pkl file.

*At this point we also store this specific model in the* **/models** *folder*

After that we can use the *.predict()* function from sklearn to do prediction using our test dataset.

When the predictions are done we store the results in .csv files located in **/predictions**

