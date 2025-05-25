# link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

import pandas as pd

def load_heart_data(path="heart.csv"):
    data = pd.read_csv(path)
    return data
