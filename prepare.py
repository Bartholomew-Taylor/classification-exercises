import pandas as pd
from scipy import stats
from pydataset import data
import numpy as np
import env
import os
import prepare
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer




def train_val_test(df):
    seed = 42
    train, val_test = train_test_split(df, train_size = 0.7, random_state = seed)
    validate, test = train_test_split(val_test, train_size = 0.5, random_state = seed)
    return train, validate, test