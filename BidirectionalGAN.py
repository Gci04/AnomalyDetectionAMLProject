from keras.datasets import cifar10, mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from preprocessing import image_normalization_mapping
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import log_loss
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BIGAN():
    def __init__(self, data, NORMAL_CLASS = 8, ANOMAL_DATA_NUMB = 3000):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128

        optimizer = optimizer = RMSprop(0.0002, 0.5)
        
        self.X_train, self.Y_train, self.X_test, self.Y_test = data
        self.NORMAL_CLASS = NORMAL_CLASS
        self.ANOMAL_DATA_NUMB = 3000
        
        self.train_normal, self.test_normal = self.get_normal_data()
        self.anomal = self.get_anomal_data()
        
        # CREATE DISCRIMINATOR
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])
        #CREATE GENERATOR
        self.generator = self.create_generator()
        #CREATE ENCODER
        self.encoder = self.create_encoder()

        self.discriminator.trainable = False

        # GENERATED IMAGE
        noise = Input(shape=(self.latent_dim, )) 
        gen_img = self.generator(noise)

        #ENCODED IMAGE
        img = Input(shape=self.img_shape)
        enc_img = self.encoder(img)

        fake = self.discriminator([noise, gen_img])
        valid = self.discriminator([enc_img, img])

        #WHOLE MOSWL
        self.bigan_generator = Model([noise, img], [fake, valid])
        self.bigan_generator.compile(loss='binary_crossentropy',
            optimizer=optimizer)

    def get_normal_data(self):
        idx = np.where(self.Y_train[:,0]==self.NORMAL_CLASS)
        train_normal = self.X_train[idx]
        idx = np.where(self.Y_test[:,0]==self.NORMAL_CLASS)
        test_normal = self.X_test[idx]
        train_normal = image_normalization_mapping(train_normal, 0, 255, -1, 1)
        test_normal = image_normalization_mapping(test_normal, 0, 255, -1, 1)
        return train_normal, test_normal
    def get_anomal_data(self):
        idx = np.where(self.Y_train[:,0]!=self.NORMAL_CLASS)[0][:self.ANOMAL_DATA_NUMB]
        anomal = self.X_train[idx]
        anomal = image_normalization_mapping(anomal, 0, 255, -1, 1)
        return anomal

    def create_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def create_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)
        
        return Model(z, gen_img)

    def create_discriminator(self):
        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)
    
    def Batches(self,normal_data, batch_size):
        N = normal_data.shape[0]
        current_idx = 0
        while current_idx < N:
            start_idx = current_idx
            end_idx = start_idx + batch_size
            if end_idx > N:
                end_idx = N
            data = normal_data[start_idx:end_idx]
            current_idx += batch_size
            yield data
    def train(self, epochs, batch_size=128, interval=10):

        #LOSSES TO PLOT
        self.G_LOSS = []
        self.D_LOSS = []
        N = self.train_normal.shape[0]/batch_size
        for epoch in range(epochs):
            G_LOSS_ = 0
            D_LOSS_ = 0
            for normal in self.Batches(self.train_normal, batch_size):
                
                #LABELS
                valid = np.ones((normal.shape[0], 1))
                fake = np.zeros((normal.shape[0], 1))
                
                #TRAIN DISCRIMINATOR
                noise = np.random.normal(size=(normal.shape[0], self.latent_dim))
                fakes = self.generator.predict(noise)
                
                encoded = self.encoder.predict(normal)
                
                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([encoded, normal], valid)
                d_loss_fake = self.discriminator.train_on_batch([noise, fakes], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                #TRAIN GENERATOR    
                # Train the generator (z -> img is valid and img -> z is invalid)
                g_loss = self.bigan_generator.train_on_batch([noise, normal], [valid, fake])
                G_LOSS_ += g_loss[0]
                D_LOSS_ += d_loss[0]
            
                
            self.G_LOSS.append(G_LOSS_/N)
            self.D_LOSS.append(D_LOSS_/N)
            
            # If at save interval => save generated image samples
            #if epoch % interval == 0:
                #self.visualize(epoch)
                #print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, D_LOSS_/N, 100*d_loss[1], G_LOSS_/N))

    def visualize(self, epoch):
        z = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(z)

        gen_imgs = image_normalization_mapping(gen_imgs, -1, 1, 0, 255).astype('uint8')        
        plt.imshow(gen_imgs[0])
        plt.show()
    
    
    def calculate_anomal_score(self, image, alpha = 0.9):
        image = image_normalization_mapping(image, 0, 255, -1, 1)
        encoded = self.encoder.predict(image[np.newaxis,:])
        Loss_G = image - self.generator.predict(encoded)
        Loss_G = np.ndarray.flatten(Loss_G)
        Loss_G = norm(Loss_G, 1)#np.sum(np.abs(Loss_G))/(np.product(image.shape))
        Loss_D = self.discriminator.predict([encoded, image[np.newaxis,:]])[0][0]
        Loss_D = -np.log(Loss_D)
        
        anomal_score = alpha*Loss_G + (1-alpha)*Loss_D
        return anomal_score
    
    def anomal_scores(self, data):
      self.scores = []
      for d in data:
        self.scores.append(self.calculate_anomal_score(d))
      return self.scores

      

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    data = X_train, Y_train, X_test, Y_test
    
    precis_1, precis_0 = [], []
    recall_1, recall_0 = [], []
    f1_score_1, f1_score_0 = [], []
    accuracy = []
    TRAIN_TIME = []
    PREDICT_TIME = []
    
    for i in range(1):
      bigan = BIGAN(data, NORMAL_CLASS = i)
      start = time.time()
      bigan.train(epochs=500, batch_size=512)
      end = time.time()
      TRAIN_TIME.append(end - start)
      test_mixed = np.concatenate((bigan.anomal, bigan.test_normal))
      start = time.time()
      scores = bigan.anomal_scores(test_mixed)
      end = time.time()
      PREDICT_TIME.append(end - start)
      labels = np.concatenate((np.zeros(bigan.anomal.shape[0]), np.ones(bigan.test_normal.shape[0])))
      prediction_idx = np.argsort(scores)[-bigan.anomal.shape[0]:]
      prediction = np.ones(bigan.anomal.shape[0]+bigan.test_normal.shape[0])
      prediction[prediction_idx] = 0

      precis_1.append(precision_score(labels, prediction,  average='macro'))
      precis_0.append(precision_score(labels, prediction,  pos_label=0))
      recall_1.append(recall_score(labels, prediction,  average='macro'))
      recall_0.append(recall_score(labels, prediction,  pos_label=0))
      f1_score_1.append(f1_score(labels, prediction,  average='macro'))
      f1_score_0.append(f1_score(labels, prediction,  pos_label=0))
      accuracy.append(accuracy_score(labels, prediction))
      
      
      
    #PRECISION ON AVERAGE FOR CLASS 0
    print("mean_prec0 = ", np.mean(precis_0))
    print("std_prec0 = ", np.std(precis_0))
    #PRECISION ON AVERAGE FOR CLASS 1
    print("mean_prec1 = ", np.mean(precis_1))
    print("std_prec1 = ", np.std(precis_1))
    
    #RECALL ON AVERAGE FOR CLASS 0
    print("mean_rec0 = ", np.mean(recall_0))
    print("std_rec0 = ", np.std(recall_0))
    
    #RECALL ON AVERAGE FOR CLASS 1
    print("mean_rec1 = ", np.mean(recall_1))
    print("std_rec1 = ", np.std(recall_1))
    
    #RECALL ON AVERAGE FOR CLASS 0
    print("mean_f10 = ", np.mean(f1_score_0))
    print("std_f10 = ", np.std(f1_score_0))
    
    #RECALL ON AVERAGE FOR CLASS 1
    print("mean_f11 = ", np.mean(f1_score_1))
    print("std_f11 = ", np.std(f1_score_1))
    
    print("accuracy = ", np.mean(accuracy))
    print("accuracy = ", np.std(accuracy))
