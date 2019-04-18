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
import pickle
# from preprocessing import image_normalization_mapping

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
def image_normalization_mapping(image, from_min, from_max, to_min, to_max):
    """
    Map data from any interval [from_min, from_max] --> [to_min, to_max]
    Used to normalize and denormalize images
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)
train_normal = image_normalization_mapping(train_normal, 0, 255, -1, 1)
test_normal = image_normalization_mapping(test_normal, 0, 255, -1, 1)
anomal = image_normalization_mapping(anomal, 0, 255, -1, 1)

def get_mse(original, decoded):
    error = np.sum((original - decoded) ** 2)
    error /= IM_SIZE * IM_SIZE
    return error


def Convolutional_Autoencoder(lr):
    input_img = Input(shape = (IM_SIZE,IM_SIZE,IM_CHANNELS))
    l1 = Conv2D(filters=16, kernel_size=2, strides=(2,2), input_shape=(IM_SIZE,IM_SIZE,IM_CHANNELS), activation = 'tanh')(input_img)
    l2 =  Conv2D(filters=64, kernel_size=2, strides=(2,2), activation = 'tanh')(l1)
    l3 = Conv2D(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l2)
    l4 =  Conv2D(filters=256, kernel_size=2, strides=(2,2), activation = 'tanh')(l3)

    encoding = Conv2D(filters=32, kernel_size=2, strides=(2,2), activation = 'tanh')(l4)

    l5 = Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2), activation = 'tanh')(encoding)
    l6 = Conv2DTranspose(filters=256, kernel_size=2, strides=(2,2), activation = 'tanh')(l5)
    l7 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l6)
    l8 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2), activation = 'tanh')(l7)

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
    optimizer = Adam(lr, 0.5)

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

def train(epochs=5000, lr = 0.001):
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

        if epoch%100==0:
          img = image_normalization_mapping(train_normal[2], -1, 1, 0, 255).astype('uint8')
          plt.imshow(img)
          plt.show()
          img = autoencoder.predict(train_normal[2][np.newaxis,:])
          img = image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
          plt.imshow(img)
          plt.show()
          img = image_normalization_mapping(train_normal[0], -1, 1, 0, 255).astype('uint8')
          plt.imshow(img)
          plt.show()
          img = autoencoder.predict(train_normal[0][np.newaxis,:])
          img = image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
          plt.imshow(img)
          plt.show()
        print(f'epoch {epoch} loss: {LOSS} accuracy: {ACC} mse: {MSE}')

    img = image_normalization_mapping(train_normal[2], -1, 1, 0, 255).astype('uint8')
    plt.imshow(img)
    plt.show()
    img = autoencoder.predict(train_normal[2][np.newaxis,:])
    img = image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
    plt.imshow(img)
    plt.show()
    return dim_reducer


dim_reducer = train()

#SAVE MODEL
with open('dim_reducer.pickle', 'wb') as f:
  pickle.dump(dim_reducer, f)

#REDUCE DIMENTION FOR EACH IMAGE
reduced_anomal = dim_reducer.predict(anomal)
print(reduced_anomal.shape)
reduced_train_normal = dim_reducer.predict(train_normal)
reduced_test_normal = dim_reducer.predict(test_normal)
test_mixed = np.concatenate((reduced_test_normal,reduced_anomal))
labels = np.concatenate((np.ones(reduced_test_normal.shape[0]),np.zeros(reduced_anomal.shape[0])))

#SAVE IMAGES WITH REDUCED DIMENTIONS
with open('train_normal.pickle', 'wb') as f:
  pickle.dump(reduced_train_normal, f)

with open('test_mixed.pickle', 'wb') as f:
  pickle.dump(test_mixed, f)

with open('labels.pickle', 'wb') as f:
  pickle.dump(labels, f)
