# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:31:28 2019

@author: Legen
"""

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



######## TAPAYA HUA CODE ##########################################3
def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def build_filters():
 filters = []
 ksize = 9
 for theta in np.arange(0, np.pi, np.pi / 8):
  for lamda in np.arange(0, np.pi, np.pi/4): 
   kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
   kern /= 1.5*kern.sum()
   filters.append(kern)
 return filters
 
def process(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
  #fimg = cv2.filter2D(img,-1, kern)
  fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
  np.maximum(accum, fimg, accum)
 return accum


######################################################################33###############3#


lbp_features = []
for image in Data:
    fd = local_binary_pattern(image, 8, 2, method='default')
    lbp_features.append(fd)

for i in range(len(lbp_features)):
    lbp_features[i] = lbp_features[i].flatten()

######################################################################################3






def applyPCA():
    pca = PCA(0.99)
    pca.fit(traindata)
    train_img = pca.transform(traindata)
    test_img = pca.transform(testdata)

def appyLDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components=10)  
    X_train = lda.fit_transform(traindata,trainlabel)  
    X_test = lda.transform(testdata)
    
    
    
    
def accuracy_score(testlabel,pred):
    acc=0
    for i in range(len(testlabel)):
        if testlabel[i]==pred[i]:
            acc=acc+1
    acc = acc/float(len(testlabel))
    return acc



def logisticReg(traindata,testdata,trainlabel,testlabel):
    #lg = LogisticRegressionCV(multi_class="ovr").fit(traindata,trainlabel)
    #lg = OneVsRestClassifier(LinearSVC(random_state=0)).fit(traindata,trainlabel)
    lg = RandomForestClassifier(n_estimators=200, random_state=50).fit(traindata,trainlabel)
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



with open('./fer2013/fer2013.csv', 'r') as csvfile:
     csvreader = csv.reader(csvfile)
     fields = next(csvreader)
     total = []
     Data = []
     label = []
     typeData = []
     for row in csvreader:
         total.append(row)
     for row in range(len(total)):
         label.append(int(total[row][0]))
         Data.append(np.array(list(map(int,total[row][1].split(" ")))))
         typeData.append(total[row][2])
     for i in range(len(Data)):
         Data[i] = np.reshape(Data[i],(48,48))  
         
###############HOG FEATURES########################
ppc = 8
hog_features = []
for image in Data:
    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualise=True)
    hog_features.append(fd)
######################################################3        
         
traindata,testdata,trainlabel,testlabel = train_test_split(hog_features,
                                                       label,
                                                         test_size=0.20,random_state=42)  

best = fivefoldcross(traindata,trainlabel)
prob = best.predict_proba(testdata)                                                   
pred = best.predict(testdata)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)


face_cascade = cv2.CascadeClassifier('C:\\Users\\Legen\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(tryim,minSize=(10,10))
for face in faces:
    x1, y1, w, h = face
    x2 = x1 + w
    y2 = y1 + h
    
im=Data[0];

im=np.array(im,np.uint8)

tryimg = cv2.resize(im,(240,240))  



lab =  ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
print(metrics.classification_report(testlabel, pred,
         target_names=lab))

lab =  ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
cm = confusion_matrix(pred,testlabel,8)
seaborn.heatmap(cm,annot=True,fmt="d",xticklabels=lab,yticklabels=lab,linewidths=.5)
plt.title('Confusion Marix_HOG+PCA_0.99+SVM')
plt.xlabel('Predicted Class ---->')
plt.ylabel('True Class ---->')
plt.show()   

lab =  ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
C1prob = prob
truelabel = testlabel
for f in range(8):
     truelab = [x for _,x in sorted(zip(C1prob[:,f],truelabel),reverse=True)]
     for i in range(len(truelab)):
       if truelab[i]!=f:
          truelab[i]=1
       else:
          truelab[i]=0
     predlabel=[]
     for i in range(len(truelab)):
        predlabel.append(1)
     FPR=[]
     TPR=[]
    
     for i in range(len(truelab)):
        cm=confusion_matrix1(truelab,predlabel)
        #print(cm)
        a=cm[1][0]/float(cm[1][1]+cm[1][0])
        FPR.append(a)
        b=cm[0][0]/float(cm[0][1]+cm[0][0])
        TPR.append(b)
        predlabel[i]=0
    
     plt.plot(FPR,TPR,label = lab[f]) 
plt.xlabel('FPR----->')
plt.ylabel('TPR----->') 
plt.legend()
plt.title('ROC_Curve_HOG+PCA_0.99+RF') 
plt.show()