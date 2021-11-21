Dataset informations
====================

For this project we have been working on a dataset from kaggle called "Home Credit Default Risk" that can be found at :

https://www.kaggle.com/c/home-credit-default-risk

The dataset contains various informations about previous loans and the if the loans has been repayed by the borrower.

Files 
-------

We were working on two separates files :

* *application_train* that contains numerous data about previous loans and a **Target** feature that will be use to train our model and that represents the ability to repay the loan.

* *application_test* that also contains data about previous loans but without the target feature, it will be used by our models to make predictions.

Importing the files 
---------------------

We didn't implement a way to automatically import the files from kaggle.

To acces the data, we simply downloaded them from kaggle and placed them in the data/raw folder

