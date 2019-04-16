import numpy as np
import tensorflow as tf
sess = tf.Session()
from collections import defaultdict

import keras
from keras import backend as K
K.set_session(sess)

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras.optimizers import Adam,Adagrad

from keras.utils import to_categorical
from keras.callbacks import LambdaCallback

import matplotlib.pyplot as plt
%matplotlib inline

from keras.utils.generic_utils import get_custom_objects


class OC_NN:
    def __init__:
        self.r = None
        self.hiddenLayerSize = None


    def image_to_feature_vector(image, height,width):
        #flatten the the image to 1d list/array
        return np.reshape(image,(len(image),height*width))

    def custom_loss(self,w1,w2,nu):
        def custom_hinge(y_true, y_pred):
            term1 = 0.5 * tf.reduce_sum(w1[0] ** 2)
            term2 = 0.5 * tf.reduce_sum(w2[0] ** 2)
            term3 = 1 / nu * K.mean(K.maximum(0.0, self.r - tf.reduce_max(y_pred, axis=1)), axis=-1)
            term4 = -1*self.r
            # yhat assigned to r
            self.r = tf.reduce_max(y_pred, axis=1)
            # r = nuth quantile
            self.r = tf.contrib.distributions.percentile(self.r, q=100 * nu)

            return (term1 + term2 + term3 + term4)

        return custom_hinge
    def fit(Xtrain,epochs,nu,n_classes,learning_rate):
        #create custom activation function
        def custom_activation(x):
            return (1 / np.sqrt(self.hiddenLayerSize)) * tf.cos(x / 0.02)

        get_custom_objects().update({'custom_activation':Activation(custom_activation)})

        n_samples,img_height,img_width = Xtrain.shape
        xtrain = image_to_feature_vector(Xtrain,img_height,img_width)
        #for keras custom loss function requirements
        ytrain = to_categorical(np.ones(n_samples),num_classes=n_classes)

        input_dim = img_width * img_height

        model = Sequential()
        model.add(Dense(100,input_dim = input_dim,kernel_initializer="glorot_normal",name="l1"))
        model.add(Activation(custom_activation))

        model.add(Dense(classes,name="l2"))
        model.add(Activation("linear"))


        #create an optimizer (Adam)
        optimizer = Adam(lr=learning_rate, decay=learning_rate / epochs)
        #compile model and also use custom loss function
        # model.compile(optimizer = optimizer, loss= OC_NN.custom_loss())
