
import pandas as pd # import pandas_profiling
import numpy as np
import pickle,warnings
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
#from preprocessing.py
from preprocessing import getKdd99_AE,get_kdd_data,get_isof_data
import seaborn as sn
from matplotlib import pyplot as plt
%matplotlib inline

warnings.filterwarnings('ignore')


#Mean mean_squared_error for calculating reconstruction error
def mse(pred,true):
    result = []
    for sample1,sample2 in zip(pred,true):
        error = np.sum((sample1.astype("float") - sample2.astype("float")) ** 2)
        error /= float(len(sample2))
        result.append(error)
    return np.array(result)

#reconctruction losses (AE)
def get_losses(model, x):
    reconstruct_err = []
    pred = model.predict(x)
    err = mse(x, pred)
    return err

#measure model performance in terms of accuracy, confusion_matrix and F1 score
def performance(true,pred,title="confusion matrix"):
    acc = accuracy_score(pred,true)
    print("Accurary : ",acc)
    f1 = f1_score(true,pred)
    print("F1 Score : ",f1)
    # print("Confusion matrix")
    # print(confusion_matrix(true,pred))
    print("Classification report")
    print(classification_report(pred,true))
    df_cm = pd.DataFrame(confusion_matrix(true,pred), index = ["normal","Anomal"],
                      columns = ["Normal","Anomal"])
    plt.figure(figsize = (10,7))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title,fontsize=20)
    fig = sn.heatmap(df_cm,fmt='g',annot=True,annot_kws={"size": 20})
    # plt.savefig("test.png")

def IsAnomaly(model,x,threshold=0.252):
    pred = model.predict(x)
    MSE = mse(pred,x)
    res = np.where(MSE < threshold,1,-1) #anomaly : -1, normal : 1
    return res

# ress = IsAnomaly(autoencoder,nomalies_train,0.252)
def get_anomaly_score(model,x):
    pred = model.predict(x)
    return mse(pred,x)[:,np.newaxis]

#Get train and test data
# nomalies_train, original_train, test_data, test_data_original, ytest = get_kdd_data(with_original=True)
nomalies_train, test_data, ytest = get_kdd_data()

#create an autoencoder to be also used in dimension reduction
autoencoder , dim_reducer = getKdd99_AE(nomalies_train)
#CPU times: user 6min 6s, sys: 26.6 s, total: 6min 33s
#Wall time: 4min 56s : for training AE

#reduce dimension using encoder from autoencoder (train and test set)
X_dim_reduced = dim_reducer.predict(nomalies_train)
xtext = dim_reducer.predict(test_data)

#CPU times: user 2h 32min 2s, sys: 19.9 s, total: 2h 32min 22s
# Wall time: 2h 32min 20s , nu=0.03

model = svm.OneClassSVM(kernel='rbf', nu=0.01,gamma="auto")
%%time
model.fit(X_dim_reduced)

# CPU times: user 1h 43min 19s, sys: 10.7 s, total: 1h 43min 30s
# Wall time: 1h 43min 29s
# OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma='auto', kernel='rbf',
#       max_iter=-1, nu=0.005, random_state=None, shrinking=True, tol=0.001,
#       verbose=False)
# Accurary :  0.9947191973744047
# F1 Score :  0.9923094299326434

#predict anomalies from test data
%%time
pred = model.predict(xtext)
xtext.shape
#Display accuraccy, F1 score, confusion_matrix and classification report
performance(ytest,pred,"OCSVM Confusion Matrix")

#Autoencoder as clasifier
#find the threshold
losses = get_losses(autoencoder, nomalies_train)
loss_df = pd.DataFrame(losses,columns=["loss"])
# loss_df.describe()

#test with autoencoder
#predict anomalies using autoencoder (utilizing the reconstruction error threshold) & measure performance
pred = IsAnomaly(autoencoder,test_data)
performance(ytest,pred)

# save the classifier
with open('oncsvm.pkl', 'wb') as fid:
    pickle.dump(model, fid)

# load classifier
with open('oncsvm.pkl', 'rb') as fid:
    model = pickle.load(fid)

#IsolationForest
from sklearn.ensemble import IsolationForest

X_train,xtext,ytest = get_isof_data()

isofor = IsolationForest(n_jobs=-1,n_estimators=100, behaviour="new",max_samples=256, contamination=0.6)
isofor.fit(X_train)

pred = isofor.predict(xtext)
pred = np.where(pred == -1,0,1)
performance(ytest,pred)
