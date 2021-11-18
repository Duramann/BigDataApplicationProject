import pandas as pd
import numpy as np

test_df = pd.read_csv('../../data/raw/raw_test_data.csv')
train_df = pd.read_csv('../../data/raw/raw_train_data.csv')

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

train_df.dropna(axis = 0, how = 'any', thresh = int(train_df.shape[1]*0.8),inplace=True)
test_df.dropna(axis = 0, how = 'any', thresh = int(test_df.shape[1]*0.8),inplace=True)

qualitative_c = test_df.select_dtypes(include=[object]).columns

for col in qualitative_c:
    train_df[col] = train_df[col].fillna(train_df[col].mode(dropna=True)[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode(dropna=True)[0])


quantitative_c = test_df.select_dtypes(include=[int,float]).columns

for col in quantitative_c:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

train_df.to_csv('../../data/interim/cleaned_train_data.csv')
test_df.to_csv('../../data/interim/cleaned_test_data.csv')

train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

target = train_df['TARGET']

train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)

train_df['TARGET'] = target

train_df.to_csv('../../data/processed/train_data.csv')
test_df.to_csv('../../data/processed/test_data.csv')