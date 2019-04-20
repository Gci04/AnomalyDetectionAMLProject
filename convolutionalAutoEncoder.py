# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:26:07 2019

@author: Rufina
"""
import numpy as np
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.models import  Model
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.optimizers import Adam
import pickle,time
from preprocessing import image_normalization_mapping
import time

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:26:07 2019

@author: Rufina
"""
import numpy as np
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.models import  Model
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.optimizers import Adam
import pickle
from preprocessing import image_normalization_mapping
import time

class ConvolutionalAutoencoder():
    def __init__(self, DATA, NORMAL_CLASS = 0, ANOMAL_DATA_NUMB = 3000, lr = 0.001):
        self.X_train, self.Y_train, self.X_test, self.Y_test = DATA
        self.NORMAL_CLASS = NORMAL_CLASS
        self.ANOMAL_DATA_NUMB = ANOMAL_DATA_NUMB
        self.IM_SIZE = X_train.shape[1]
        self.IM_CHANNELS = X_train.shape[3] 
        
        self.train_normal, self.test_normal = self.get_normal_data()
        self.anomal = self.get_anomal_data()
        
        self.autoencoder, self.dim_reducer = self.CreateAutoencoder(lr)
        
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
    
    def CreateAutoencoder(self, lr = 0.001):
        input_img = Input(shape = (self.IM_SIZE,self.IM_SIZE,self.IM_CHANNELS))
        l1 = Conv2D(filters=16, kernel_size=2, strides=(2,2), input_shape=(self.IM_SIZE,self.IM_SIZE,self.IM_CHANNELS), activation = 'tanh')(input_img)
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
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    
        dim_reducer = Model(inputs = input_img, outputs = encoding)
    
        return autoencoder, dim_reducer
    
    def Batches(self,normal_data, batches_numb):
        N = normal_data.shape[0]
        batch_size = int(N/batches_numb) + 1
        for current_batch in range(batches_numb):
            start_idx = current_batch * batch_size
            end_idx = start_idx + batch_size
            if end_idx > N:
                end_idx = N
            data = normal_data[start_idx:end_idx]
            yield data
    
    def visualize(self):
        img = image_normalization_mapping(self.train_normal[2], -1, 1, 0, 255).astype('uint8')
        plt.imshow(img)
        plt.show()
        img = self.autoencoder.predict(self.train_normal[2][np.newaxis,:])
        img = image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
        plt.imshow(img)
        plt.show()
        img = image_normalization_mapping(self.train_normal[0], -1, 1, 0, 255).astype('uint8')
        plt.imshow(img)
        plt.show()
        img = self.autoencoder.predict(self.train_normal[0][np.newaxis,:])
        img = image_normalization_mapping(img[0], -1, 1, 0, 255).astype('uint8')
        plt.imshow(img)
        plt.show()
    
    def train(self,epochs=1000):
        for epoch in range(epochs):
            n = 0
            LOSS, ACC = 0., 0.
            for normal_data in self.Batches(self.train_normal, 1):
                loss, accuracy = self.autoencoder.train_on_batch(normal_data,normal_data)
#                 n+=1
#                 LOSS += loss
#                 ACC += accuracy
#             LOSS/= n
#             ACC/=n
#             if epoch%100==0:
#                 self.visualize()
#                 print(f'epoch {epoch} loss: {LOSS} accuracy: {ACC}')
<<<<<<< HEAD

=======
    
>>>>>>> fc9564729c8776483e7e793a250a6b359a2d823c
    def save_models(self):
        with open(f'dim_reducer_{self.NORMAL_CLASS}.pickle', 'wb') as f:
            pickle.dump(self.dim_reducer, f)
        with open(f'autoencoder_{self.NORMAL_CLASS}.pickle', 'wb') as f:
<<<<<<< HEAD
            pickle.dump(self.autoencoder, f)

=======
            pickle.dump(self.autoencoder, f)   
            
>>>>>>> fc9564729c8776483e7e793a250a6b359a2d823c
    def reduce_dim_and_save(self):
        #REDUCE DIMENTION FOR EACH IMAGE
        reduced_anomal = self.dim_reducer.predict(self.anomal)
        reduced_train_normal = self.dim_reducer.predict(self.train_normal)
        reduced_test_normal = self.dim_reducer.predict(self.test_normal)
        test_mixed = np.concatenate((reduced_test_normal,reduced_anomal))
        labels = np.concatenate((np.ones(reduced_test_normal.shape[0]),np.zeros(reduced_anomal.shape[0])))
        
        #SAVE IMAGES WITH REDUCED DIMENTIONS
        with open(f'train_normal_{self.NORMAL_CLASS}.pickle', 'wb') as f:
          pickle.dump(reduced_train_normal, f)
<<<<<<< HEAD

        with open(f'test_mixed_{self.NORMAL_CLASS}.pickle', 'wb') as f:
          pickle.dump(test_mixed, f)

=======
        
        with open(f'test_mixed_{self.NORMAL_CLASS}.pickle', 'wb') as f:
          pickle.dump(test_mixed, f)
        
>>>>>>> fc9564729c8776483e7e793a250a6b359a2d823c
        with open(f'labels_{self.NORMAL_CLASS}.pickle', 'wb') as f:
          pickle.dump(labels, f)

#RUN EVERYTHING
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
<<<<<<< HEAD
DATA = (X_train, Y_train,X_test, Y_test)
TRAIN_TIME = []

for i in range(10):
    cautoenc = ConvolutionalAutoencoder(DATA, NORMAL_CLASS = i, ANOMAL_DATA_NUMB = 3000, lr = 0.001)
    start = time.time()
    cautoenc.train()
    end = time.time()
    TRAIN_TIME.append(end - start)
    cautoenc.save_models()
    cautoenc.reduce_dim_and_save()

#SAVE TIME
with open(f'train_time.pickle', 'wb') as f:
    pickle.dump(TRAIN_TIME, f)
#RUN EVERYTHING
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

DATA = (X_train, Y_train,X_test, Y_test)
cautoenc = ConvolutionalAutoencoder(DATA, NORMAL_CLASS = 0, ANOMAL_DATA_NUMB = 3000, lr = 0.001)
cautoenc.train()
cautoenc.save_models()
cautoenc.reduce_dim_and_save()

#predict using CNN-Autoencoder
def get_mse(original, decoded):
    n = len(original[0].flatten())
    res = []
    for img1,img2 in zip(original,decoded):
        error = np.sum((img2.flatten() - img1.flatten()) ** 2)
        error /= n
        res.append(error)
    return np.expand_dims(res,1)

def predict_anomaly(model,imgs,threshold):
    pred = model.predict(imgs)
    MSE = get_mse(pred,imgs)
    res = np.where(MSE > threshold,0,1) #anomaly : 0, normal : 1
    return res
def perfomence(true,pred):
    acc = accuracy_score(pred,true)
    print("Accurary : ",acc)
    f1 = f1_score(true,pred)
    print("F1 Score : ",f1)
    print("Classification report")
    print(classification_report(pred,true))
    df_cm = pd.DataFrame(confusion_matrix(true,pred), index = ["normal","Anomal"],
                      columns = ["Normal","Anomal"])
    plt.figure(figsize = (10,7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = sn.heatmap(df_cm,fmt='g',cmap='Blues',annot=True,annot_kws={"size": 20})


#classify images using CNN autoencoder and utilizing the reconctruction error
def load_models(path):
    models = []
    for i in range(10):
        name = path+"autoencoder_"+str(i)+".pickle"
        with open(name, 'rb') as fid:
            models.append(pickle.load(fid))
    return models

res = load_models("./CNN_AE/")

# with open('./convAutoEncOutput/train_normal.pickle', 'rb') as fid:
#     X = pickle.load(fid)
# with open('./convAutoEncOutput/test_mixed.pickle', 'rb') as fid:
#     test_mixed = pickle.load(fid)

#Get threshold and get insigt on possible thresholds
train_normal, test_normal = cautoenc.get_normal_data()

t = []
for model in res:
    # start = time.time()
    pr = model.predict(h)
    # end = time.time()
    # print(end - start


    # pred = model.predict(train_normal)


err = get_mse(train_normal,pred)
losses = pd.DataFrame(err,columns=["loss"])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)


_ = ax.hist(losses.loss.values, bins=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Frequency",fontsize=16,fontweight='bold')
plt.xlabel("Mean Squared Error",fontsize=16,fontweight='bold')
fig.savefig("CNN_AE_losses.png")
plt.show()
# losses.describe()
#find threshold
t = []
for i in np.linspace(0.015,0.3,50):
    pred = predict_anomaly(cnn_autoencoder,tes,i)
    t.append((i,accuracy_score(labels,pred),f1_score(labels,pred)))

threshold_df = pd.DataFrame(t,columns=["Threshold","Accuracy","F1 score"])


#create labels for test data. Normal: 1, Anomal: 0
y_anomal = np.ones((len(anomal),1))
y_normal = np.zeros((len(test_normal),1))
labels = np.concatenate((y_normal,y_anomal))

#test data
xtest = np.concatenate((test_normal,anomal),axis=0)

#predict and measure performance
pred = predict_anomaly(cnn_autoencoder,xtest,0.071)
perfomence(pred,labels)
=======
DATA = (X_train, Y_train,X_test, Y_test)
TRAIN_TIME = []

for i in range(10):
    cautoenc = ConvolutionalAutoencoder(DATA, NORMAL_CLASS = i, ANOMAL_DATA_NUMB = 3000, lr = 0.001)
    start = time.time()
    cautoenc.train()
    end = time.time()
    TRAIN_TIME.append(end - start)
    cautoenc.save_models()
    cautoenc.reduce_dim_and_save()
    
#SAVE TIME
with open(f'train_time.pickle', 'wb') as f:
    pickle.dump(TRAIN_TIME, f)  
>>>>>>> fc9564729c8776483e7e793a250a6b359a2d823c
