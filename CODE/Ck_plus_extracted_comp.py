# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:17:36 2019

@author: Legen
"""

import os
import cv2
import numpy as np
import csv
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import math
#import matplotlib.pyplot.subplot
import matplotlib.pyplot as plt
import seaborn
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.feature import canny
from skimage.feature import blob_log
from sklearn.utils import shuffle
from sklearn import svm
import seaborn

def accuracy_score(testlabel,pred):
    acc=0
    for i in range(len(testlabel)):
        if testlabel[i]==pred[i]:
            acc=acc+1
    acc = acc/float(len(testlabel))
    return acc

def confusion_matrix(predlabel,truelabel,clas):
    cm=[]
    for i in range(clas):
        A=[]
        for j in range(clas):
            A.append(0)
        cm.append(A)
    
    for i in range(len(predlabel)):
        cm[truelabel[i]][predlabel[i]]=cm[truelabel[i]][predlabel[i]]+1
            
    return cm


def logisticReg(traindata,testdata,trainlabel,testlabel):
    #lg = LogisticRegressionCV(multi_class="ovr").fit(traindata,trainlabel)
    #lg = OneVsRestClassifier(LinearSVC(random_state=0)).fit(traindata,trainlabel)
    lg = svm.SVC(gamma=0.10,kernel = "rbf" ,probability=True, C=10.).fit(traindata,trainlabel)
    #lg = RandomForestClassifier(n_estimators=200, random_state=50).fit(traindata,trainlabel)
    #lg = GaussianNB().fit(traindata,trainlabel)
    pred = lg.predict(testdata)
    acc = accuracy_score(testlabel,pred)
    print("The Accuracy is")
    print(acc)
    return (acc,lg)



def findmean_std(acmat):
    mean=np.average(np.array(acmat))
    sd=np.std(acmat)
    print ("Mean= "+str(mean))
    print ("Standard Deviation= "+str(sd))
    
    
def fivefoldcross(traindata,trainlabel):
    low=0
    k=2
    temp=len(traindata)/5
    high=temp
    acmat=[]
    train = []
    for i in range(5):
        newtraindata=[]
        newtestdata=[]
        newtrainlabel=[]
        newtestlabel=[]
        for j in range(len(traindata)):
            if j>=low and j<high:
                newtestdata.append(traindata[j])
                newtestlabel.append(trainlabel[j])
            else:
                newtraindata.append(traindata[j])
                newtrainlabel.append(trainlabel[j])
        print("training "+str(k-1)+" started")
        ac=logisticReg(newtraindata,newtestdata,newtrainlabel,newtestlabel)
        acmat.append(ac[0])
        train.append(ac[1])
        low=high
        high=k*temp
        k=k+1
    print(acmat)
    findmean_std(acmat)
    ind = acmat.index(np.max(acmat))
    return train[ind]






def applyPCA(traindata):
    pca = PCA(n_components = 30)
    pca.fit(traindata)
    train_img = pca.transform(traindata)
    return train_img

def hog_features(Data):
    ppc = 8
    hog_features = []
    for image in Data:
        fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
        hog_features.append(fd)
    return hog_features

##################################################################
label=[]
DataUrl=[]
lab=0
folders = os.listdir("./Dataset_Crop_CK")
for folder in folders:
    file = os.listdir("./Dataset_Crop_CK/"+folder)
    for i in range(0,len(file),4):
        img = []
        img.append("./Dataset_Crop_CK/"+folder+"/"+file[i])
        img.append("./Dataset_Crop_CK/"+folder+"/"+file[i+1])
        img.append("./Dataset_Crop_CK/"+folder+"/"+file[i+2])
        img.append("./Dataset_Crop_CK/"+folder+"/"+file[i+3])
        DataUrl.append(img)
        label.append(lab)
    lab = lab+1

####################################################################

Data = []   
left_eye = []
lip = []
middle = []
right_eye =[] 

for block in DataUrl:
        A = []
        B = []
        C = []
        D = []
        A.append(cv2.resize(cv2.imread(block[0],0),(int(48),int(48))))
        B.append(cv2.resize(cv2.imread(block[1],0),(int(48),int(48))))
        C.append(cv2.resize(cv2.imread(block[2],0),(int(48),int(48))))
        D.append(cv2.resize(cv2.imread(block[3],0),(int(48),int(48))))
        left_eye.append(A[0])
        lip.append(B[0])
        middle.append(C[0])
        right_eye.append(D[0])

'''
for i in range(len(left_eye)):
    left_eye[i] = left_eye[i].ravel()
    lip[i] = lip[i].ravel()
    middle[i] = middle[i].ravel()
    right_eye[i] = right_eye[i].ravel()
'''

hog_left_eye = hog_features(left_eye)
pca_left_eye = applyPCA(hog_left_eye)

hog_lip = hog_features(lip)
pca_lip = applyPCA(hog_lip)

hog_middle = hog_features(middle)
pca_middle = applyPCA(hog_middle)

hog_right_eye = hog_features(right_eye)
pca_right_eye = applyPCA(hog_right_eye)      
        
       
Data = np.concatenate((pca_left_eye,pca_lip,pca_middle,pca_right_eye),axis=1 )

temp_data, temp_label = shuffle ((Data), (label)) 
traindata,testdata,trainlabel,testlabel = train_test_split(temp_data,
                                                       temp_label,
                                                         test_size=0.20,random_state=42)  

best = fivefoldcross(traindata,trainlabel)
pred = best.predict(testdata)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)

cm = confusion_matrix(pred,testlabel,8)
seaborn.heatmap(cm,annot=True,fmt="d",linewidths=.5)
plt.title('Confusion Marix_LDA')
plt.xlabel('Predicted Class ---->')
plt.ylabel('True Class ---->')
plt.show()

       
        


        
        
        