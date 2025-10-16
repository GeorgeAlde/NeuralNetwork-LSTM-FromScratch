import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
print(train.isnull().sum())
train = train.fillna({"Embarked": "S"})
print(train.isnull().sum())
print(train.mean(numeric_only = True))
