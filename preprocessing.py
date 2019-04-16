#!curl -o data.gz -L "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
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
    

def image_normalization_mapping(image, from_min, from_max, to_min, to_max):
    """
    Map data from any interval [from_min, from_max] --> [to_min, to_max]
    Used to normalize and denormalize images
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)
