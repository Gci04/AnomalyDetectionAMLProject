# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:26:07 2019

@author: Rufina
"""
import numpy as np
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.models import Sequential,  Model
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, ReLU, Input, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adam
from preprocessing import image_normalization_mapping

#CONSTANT
NORMAL_CLASS = 0
ANOMAL_DATA_NUMB = 3000
IM_SIZE = 32
IM_CHANNELS = 3

#LOAD THE DATA
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#GET DATA FOR NORMAL CLASS
idx = np.where(Y_train[:,0]==NORMAL_CLASS)
train_normal = X_train[idx]
idx = np.where(Y_test[:,0]==NORMAL_CLASS)
test_normal = X_test[idx]

#TAKE ANOMAL DATA
idx = np.where(Y_train[:,0]!=NORMAL_CLASS)[0][:ANOMAL_DATA_NUMB]
anomal = X_train[idx]

#NORMALIZE DATA

train_normal = image_normalization_mapping(train_normal, 0, 255, -1, 1)
test_normal = image_normalization_mapping(test_normal, 0, 255, -1, 1)
anomal = image_normalization_mapping(anomal, 0, 255, -1, 1)

def get_mse(original, decoded):
    error = np.sum((original - decoded) ** 2)
    error /= IM_SIZE * IM_SIZE
    return error

def Convolutional_Autoencoder(lr):
    input_img = Input(shape = (IM_SIZE,IM_SIZE,IM_CHANNELS))
    l1 = Conv2D(filters=3, kernel_size=2, strides=(2,2), input_shape=(IM_SIZE,IM_SIZE,IM_CHANNELS), activation = 'tanh')(input_img)
    l2 =  Conv2D(filters=32, kernel_size=2, strides=(2,2), activation = 'tanh')(l1)
    l3 = Conv2D(filters=64, kernel_size=2, strides=(2,2), activation = 'tanh')(l2)
    l4 =  Conv2D(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l3)
    
    encoding = Conv2D(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l4)
    
    l5 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(encoding)
    l6 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l5)
    l7 = Conv2DTranspose(filters=64, kernel_size=2, strides=(2,2), activation = 'tanh')(l6)
    l8 = Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), activation = 'tanh')(l7)
    decoded = Conv2DTranspose(filters=3, kernel_size=2, strides=(2,2), activation = 'tanh')(l8)
    
    autoencoder = Model(inputs=input_img , outputs=decoded)    
    optimizer = Adam(lr)
    autoencoder.compile(loss='mean_squared_error', 
        optimizer=optimizer, metrics=['accuracy', get_mse])
    
    dim_reducer = Model(inputs = input_img, outputs = encoding)
    
    return autoencoder, dim_reducer

def Classical_autoencoder(lr = 0.01): 
    autoencoder = Sequential()
    autoencoder.add(Flatten())
    autoencoder.add(Dense(128, activation="tanh"))
    autoencoder.add(Dense(64, activation="tanh"))
    autoencoder.add(Dense(32, activation="tanh"))
    autoencoder.add(Dense(4,activation=None))
    
    
    autoencoder.add(Dense(4, activation="tanh"))
    autoencoder.add(Dense(32, activation="tanh"))
    autoencoder.add(Dense(64, activation='tanh'))
    
    autoencoder.add(Dense(IM_SIZE*IM_SIZE*IM_CHANNELS,activation='tanh'))
    autoencoder.add(Reshape((IM_SIZE,IM_SIZE,IM_CHANNELS)))
    
    inp = Input(shape=(IM_SIZE, IM_SIZE, IM_CHANNELS))
    out = autoencoder(inp)
    
    autoencoder_model = Model(inputs=inp , outputs=out)
    autoencoder_model.summary()
    optimizer = Adam(lr)
    
    autoencoder_model.compile(loss='mean_squared_error', 
        optimizer=optimizer, metrics=['accuracy', get_mse])
    
    return autoencoder_model

   
def Batches(normal_data, batches_numb):
    N = normal_data.shape[0]
    batch_size = int(N/batches_numb) + 1
    for current_batch in range(batches_numb):
        start_idx = current_batch * batch_size 
        end_idx = start_idx + batch_size 
        if end_idx > N:
            end_idx = N
        data = normal_data[start_idx:end_idx]
        yield data

def train(epochs=500, lr = 0.01):
    autoencoder, dim_reducer = Convolutional_Autoencoder(lr)
    for epoch in range(epochs):
        n = 0
        LOSS, ACC, MSE = 0., 0., 0.
        for normal_data in Batches(train_normal, 1):
            loss, accuracy, mse = autoencoder.train_on_batch(normal_data,normal_data)
            n+=1
            LOSS += loss
            ACC += accuracy
            MSE += mse
        LOSS/= n
        ACC/=n
        MSE/=n
        
        print(f'epoch {epoch} loss: {LOSS} accuracy: {ACC} mse: {MSE}')
    return dim_reducer

        
dim_reducer = train()
