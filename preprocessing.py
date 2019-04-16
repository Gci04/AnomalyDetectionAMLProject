!curl -o data.gz -L "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
import os
import pandas as pd
import numpy as np

def create_train_data():
    features = None
    with open('features.txt', 'r') as f:
      features = f.read().split('\n')

    attacks = None
    with open('attacks.txt', 'r') as f:
      attacks = f.read().split('\n')

    df = pd.read_csv('data.gz', compression='gzip',names=features ,sep=',')

    labels = df.label.values
    df.label = np.where(labels == "normal.",1,0)
    df = pd.get_dummies(df, prefix=["protocol_type","service","flag"])

    df.to_csv('kdd99_train.csv.gz', compression='gzip',index=False)
