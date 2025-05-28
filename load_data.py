# link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_heart_data(path="heart.csv"):
    data = pd.read_csv(path)
    return data

def data_to_int(data):
    dataset = data.copy()
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
    return dataset