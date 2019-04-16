
import pandas as pd # import pandas_profiling
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
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

df = pd.read_csv('kdd99_train.csv.gz', compression='gzip',sep=',')
# df = pd.read_csv('my_file.csv.gz', compression='gzip',sep=',')
# df.drop(["Unnamed: 0"],inplace=True,axis=1)

df.head()
anomalies = df[df["label"]==0].sample(n=372781, replace=False, random_state=43)
nomalies = df[df["label"]==1]

# len(nomalies)+len(anomalies)
# len(nomalies)

nomalies_train ,nomalies_test = train_test_split(nomalies,test_size=0.1,random_state=41)

nomalies_train = nomalies_train.drop(["label"],axis=1).values

Scaler = StandardScaler().fit(nomalies_train)

nomalies_train = Scaler.transform(nomalies_train)

input_dim = nomalies_train.shape[1]
latent_space_size = 12
K.clear_session()
input_ = Input(shape = (input_dim, ))

layer_1 = Dense(100, activation="tanh")(input_)
layer_2 = Dense(50, activation="tanh")(layer_1)
layer_3 = Dense(25, activation="tanh")(layer_2)

encoding = Dense(latent_space_size,activation=None)(layer_3)

layer_5 = Dense(50, activation="tanh")(encoding)
layer_6 = Dense(50, activation="tanh")(layer_5)
layer_7 = Dense(100, activation='tanh')(layer_6)

decoded = Dense(input_dim,activation=None)(layer_7)

autoencoder = Model(inputs=input_ , outputs=decoded)
# opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer="adam")
autoencoder.summary()

#create TensorBoard
tb = TensorBoard(log_dir="./logs/{}".format(time()),histogram_freq=0,write_graph=True,write_images=False)


# Fit autoencoder
autoencoder.fit(nomalies_train, nomalies_train,epochs=10,validation_split=0.2 ,batch_size=100,shuffle=False,verbose=1,callbacks=[tb])
# autoencoder.fit(X, X,epochs=5,validation_split=0.2 ,
#                 batch_size=128,shuffle=True,verbose=1,callbacks=[tb])

dim_reducer = Model(inputs = input_, outputs = encoding)
X_dim_reduced = dim_reducer.predict(nomalies_train)

model = svm.OneClassSVM(kernel='rbf', nu=0.005,gamma="auto")

model.fit(X_dim_reduced)
# pp = model.predict(X_dim_reduced)
# np.unique(pp,return_counts= True)

anomalies.label = np.where(anomalies.label == 0,-1,0)

test_data = pd.concat([nomalies_test,anomalies])
ytest = test_data.label.values

xtext = dim_reducer.predict(Scaler.transform(test_data.drop(["label"],axis=1).values))
pred = model.predict(xtext)

acc = accuracy_score(pred,ytest)
print("Accurary : ",acc)
f1 = f1_score(ytest,pred)
print("F1 Score : ",f1)
# confusion_matrix(ytest,pred)
print(classification_report(pred,ytest))
