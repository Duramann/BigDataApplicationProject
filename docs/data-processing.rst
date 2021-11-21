Data processing
===============

The data processing has been tried in a notebook found at :

**/notebooks/2.0-ANTD-data-preprocessing.ipnyb** 

It was then implemented in a script found at : 

**/src/features/build_features.py** 

To know how to execute the data processing script, see :ref:`Commands <data-process>`.

Data cleaning
-------------

To clean the data, we had to deal with our missing values.

We used the **missing_values_table()** program from data exploration to select the columns that had more than 59% of missing values and drop them.

.. code-block:: python
	
	def missing_values_columns(df):
	        # count the total number of missing value in the dataframe
	        missing = df.isnull().sum()

	        # Makes it a percentage
	        percent = 100 * df.isnull().sum() / len(df)
	        
	        # Make a table with the results
	        table = pd.concat([missing, percent], axis=1)
	        
	        # Rename the columns
	        table_rename = table.rename(columns = {0: 'Number of missing values',1: '% of Total Values'})
	        
	        return table_rename[table_rename['% of Total Values'] > 59].index

	train_todrop = missing_values_columns(train_df)
	test_todrop = missing_values_columns(test_df)

	if len(train_todrop)>len(test_todrop):
	    todrop = train_todrop;
	else:
	    todrop = test_todrop


	train_df.drop(todrop, axis=1, inplace=True)
	test_df.drop(todrop, axis=1, inplace=True)

We then got rid of the rows that had more than 80% of missing values :

.. code-block:: python
	
	train_df.dropna(axis = 0, how = 'any', thresh = int(train_df.shape[1]*0.8),inplace=True)
	test_df.dropna(axis = 0, how = 'any', thresh = int(test_df.shape[1]*0.8),inplace=True)

And we choose to deal with the remaining missing values by replacing the qualitative values with their mode and the quantitative values with their median.

.. code-block:: python
	
	qualitative_c = test_df.select_dtypes(include=[object]).columns

	for col in qualitative_c:
    	train_df[col] = train_df[col].fillna(train_df[col].mode(dropna=True)[0])
    	test_df[col] = test_df[col].fillna(test_df[col].mode(dropna=True)[0])


	quantitative_c = test_df.select_dtypes(include=[int,float]).columns

	for col in quantitative_c:
    	train_df[col] = train_df[col].fillna(train_df[col].median())
    	test_df[col] = test_df[col].fillna(test_df[col].median())

After those steps, we saved the datasets in the **/data/interim** folder as csv files.

Feature engineering
-------------------

For the feature engineering, we decided to just create dummies columns for every columns of our dataset using pandas *get_dummies method*

.. code-block:: python

		train_df = pd.get_dummies(train_df)
		test_df = pd.get_dummies(test_df)

		target = train_df['TARGET']

		train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)

		train_df['TARGET'] = target

We had to align both datasets to make sure we had the same number of columns in each dataset (with the feature column being in the train dataset and not in the test one).

We saved the processed datasets in **/data/processed** folder as csv files.