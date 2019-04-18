#!curl -o data.gz -L "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
import os
import pandas as pd
import numpy as np
from time import time
from keras.layers import Input, Dense,Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from keras import optimizers, regularizers, backend as K

import warnings
warnings.filterwarnings('ignore')

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

def getKdd99_AE(Xtrain):
    input_dim = Xtrain.shape[1]
    latent_space_size = 12
    K.clear_session()
    input_ = Input(shape = (input_dim, ))

    layer_1 = Dense(100, activation="tanh")(input_)
    layer_2 = Dense(50, activation="tanh")(layer_1)
    layer_3 = Dense(25, activation="tanh")(layer_2)

    encoding = Dense(latent_space_size,activation=None)(layer_3)

    layer_5 = Dense(25, activation="tanh")(encoding)
    layer_6 = Dense(50, activation="tanh")(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim,activation=None)(layer_7)

    autoencoder = Model(inputs=input_ , outputs=decoded)
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="adam")
    # autoencoder.summary()

    #create TensorBoard
    tb = TensorBoard(log_dir="./logs/{}".format(time()),histogram_freq=0,write_graph=True,write_images=False)

    # Fit autoencoder
    autoencoder.fit(Xtrain, Xtrain,epochs=10,validation_split=0.1 ,batch_size=100,shuffle=False,verbose=1,callbacks=[tb])

    #create dimension reducer
    dim_reducer = Model(inputs = input_, outputs = encoding)

    return autoencoder , dim_reducer

def get_kdd_data():
    df = pd.read_csv('kdd99_train.csv.gz', compression='gzip',sep=',')

    anomalies = df[df["label"]==0].sample(n=372781, replace=False, random_state=43)
    nomalies = df[df["label"]==1]

    # Split normal data for train and test
    nomalies_train ,nomalies_test = train_test_split(nomalies,test_size=0.2,random_state=41)

    Scaler = StandardScaler()

    #Train data (normal data only)
    nomalies_train = nomalies_train.drop(["label"],axis=1).values
    Scaler.fit(nomalies_train)
    nomalies_train = Scaler.transform(nomalies_train)

    #test data (anomaly + normal)
    anomalies.label = np.full((len(anomalies),1),-1)
    # anomalies.label = np.where(anomalies.label == 0,-1,0)

    test_data = pd.concat([nomalies_test,anomalies])
    ytest = test_data.label.values
    test_data = Scaler.transform(test_data.drop(["label"],axis=1).values)


    return nomalies_train, test_data, ytest
