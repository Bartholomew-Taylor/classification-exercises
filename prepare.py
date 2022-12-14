import pandas as pd
from scipy import stats
from pydataset import data
import numpy as np
import env
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def clean_titanic(df):
    '''
    this function cleans up the titanic db
    '''
    
    to_drop = ['class', 'embarked','deck','passenger_id', 'age']
    df.drop(columns = to_drop, inplace = True)
    df['embark_town'].fillna('Southampton',inplace = True)
    dummies = pd.get_dummies(df[['sex','embark_town']],drop_first = [True, True])
    df = pd.concat([df,dummies], axis = 1)
    return df


def train_val_test(df, target):
    '''
    this function splits up the data into sections for training,
    validating, and testing
    models
    '''
    seed = 42
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed, stratify = df[target])
    validate, test = train_test_split(val_test, train_size = 0.5, random_state = seed,
                                      stratify = val_test[target])
    return train, validate, test