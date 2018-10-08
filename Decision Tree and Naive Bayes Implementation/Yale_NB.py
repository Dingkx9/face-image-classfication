# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:09:33 2018

@author: AnupamaKesari
"""
import scipy
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import nbImp as NB
import time
import matplotlib.pyplot as plt

# READING THE DATASET
mat = scipy.io.loadmat("YaleB.mat")  #Change path of YaleB.mat according to system

dataset = np.transpose(mat['YaleB'][0][:][1])
labels = [mat['YaleB'][0][:][0][0]]*mat['YaleB'][0][:][1].shape[1]
for x in range(1, mat['YaleB'].shape[0]):
    dataset=np.concatenate((dataset,np.transpose(mat['YaleB'][x][:][1])))
    labels = labels+ [mat['YaleB'][x][:][0][0]]*mat['YaleB'][x][:][1].shape[1]
labelInt=[]    
for x in range(0, mat['YaleB'].shape[0]):
    labelInt = labelInt + [x+1]*mat['YaleB'][x][:][1].shape[1]
    
# SPLITTING TO TRAINING AND TESTING FILES
indices = np.random.permutation(dataset.shape[0])
dataset=dataset.tolist()
Dataset_size = 2414
dataset=[dataset[indices[i]] for i in range(Dataset_size)]
labelInt=[labelInt[indices[i]] for i in range(Dataset_size)]
print("Size of Dataset:", Dataset_size)
indices=np.random.permutation(Dataset_size)

t0 = time.time()
k=5
testSize=int(len(indices)/k)
test_acc=0.0
final_train_acc=0.0

for i in range(k):
    print("\nFold Number:", i+1)
    start=testSize*i
    test_idx=indices[start:start+testSize]
    training_idx1=indices[:start]
    training_idx2=indices[start+testSize:]
    training_idx=np.append(training_idx1,training_idx2)
    training_idx = training_idx.tolist()
    test_idx = test_idx.tolist()
    training=[]
    test=[]
    for x in range(len(test_idx)):
        test.append(dataset[test_idx[x]])   
    for x in range(len(training_idx)):
        training.append(dataset[training_idx[x]])
    
    trainingLabels = [labelInt[i] for i in training_idx] 
    trainingLabels= np.asarray(trainingLabels)
    trainingLabels.shape = (len(trainingLabels),1)
    testLabels = [labelInt[i] for i in test_idx] 
    testLabels= np.asarray(testLabels)
    testLabels.shape = (len(testLabels),1)
    dataset= np.asarray(dataset)
    training= np.asarray(training)
    test= np.asarray(test)
    
    # FEATURE SELECTION
    my_model = PCA(n_components= 82, svd_solver='full')
    newSet = my_model.fit_transform(training)
    #my_model.explained_variance_ratio_.cumsum()
    newTestSet = my_model.transform(test)
    
    #MODEL BUILDING
    # No Feature Selection
#    Xnew = np.hstack((training,trainingLabels))
#    XTestNew = np.hstack((test,testLabels))
#    meanSDValues = NB.meanSDofClass(Xnew)
#    predictions = NB.predict(meanSDValues, XTestNew)
#    acc=NB.accuracy(XTestNew, predictions)
#    print('Accuracy with no Feature Selection: {0}%'.format(NB.accuracy(XTestNew, predictions)))
#    
    #Best Model
    Xnew = np.hstack((newSet[:,:82],trainingLabels))
    XTestNew = np.hstack((newTestSet[:,:82],testLabels))
    meanSDValues = NB.meanSDofClass(Xnew)
    predictions = NB.predict(meanSDValues, XTestNew)
    acc=NB.accuracy(XTestNew, predictions)
    print('Best Accuracy: {0}%'.format(acc))
    test_acc+=acc

print("Average Test Accuracy over 5-Folds= ", test_acc/k)
t1 = time.time()
print (t1 - t0)

sample_size=[250,500,750, 1000, 1500, 2000, 2414]
#tt for no 1659.5430700778961 + acc = 3.319502074688797
tt=[3.403,5.824,9.27, 13.45, 24.48, 39.86,52.6278]
acc_test_plt=[ 44.8,67.2,73.2,  77.4, 82.2, 84.1, 84.522]

plt.plot(sample_size,acc_test_plt,color='red',linestyle='solid',linewidth = 2,
         marker='o',markerfacecolor='blue',markersize=6,label='Test Accuracy')
plt.xlabel('Size of Data Set')
plt.ylabel('Accuracy (%)')

plt.title('Dataset Size vs Accuracy')
plt.show()

plt.plot(sample_size,tt,color='red',linestyle='solid',linewidth = 2,
         marker='o',markerfacecolor='blue',markersize=6,label='Test Accuracy')
plt.xlabel('Size of Data Set')
plt.ylabel('Time (seconds)')

plt.title('Dataset Size (82 PCs) vs Time taken')
plt.legend()
plt.show()