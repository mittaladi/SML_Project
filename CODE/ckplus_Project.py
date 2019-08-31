# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:43:45 2019

@author: Legen
"""



import os
import cv2
import numpy as np

########################################################################################
##### READ URL AND LABEL FROM FOLDERS##################################################

label=[]
DataUrl=[]
lab=0
folders = os.listdir("./Dataset_CK+TFEID")
for folder in folders:
    file = os.listdir("./Dataset_CK+TFEID/"+folder)
    for i in file:
        DataUrl.append("./Dataset_CK+TFEID/"+folder+"/"+i)
        label.append(lab)
    lab = lab+1
##########################################################################################
##########################################################################################
Data = []   
for url in DataUrl:
    #Data.append(cv2.resize(cv2.imread(url,0),(int(200),int(200))))
    Data.append(cv2.imread(url,0))

face_cascade = cv2.CascadeClassifier('C:\\Users\\Legen\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
faceData = []
for image in Data:
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = image[y:y+h, x:x+w]
    faceData.append(roi_gray)
    
    
for i in range(len(faceData)):
    faceData[i] = cv2.resize(faceData[i],(int(200),int(200)))
    
    
for i in range(205,214):
    faces = face_cascade.detectMultiScale(Data[i])
    for (x,y,w,h) in faces:
        img = cv2.rectangle(Data[i],(x,y),(x+w,y+h),(255,0,0),2)    
    plt.subplot(3,3,i+1)
    plt.imshow(Data[205+i])
plt.show()






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

def roccurve(C1prob,truelabel):
     
     
     for f in range(11):
         
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
    
         plt.plot(FPR,TPR) 
     plt.xlabel('FPR----->')
     plt.ylabel('TPR----->') 
     plt.title('ROC Curve') 
     plt.show()

def confusion_matrix1(truelabels,predlabels):
    cm=[[0,0],[0,0]]
 
    for i in range(len(truelabels)):
        if truelabels[i]==0 and predlabels[i]==0:
            cm[0][0]=cm[0][0]+1
            
        elif truelabels[i]!=0 and predlabels[i]==0:
            cm[1][0]=cm[1][0]+1
            
        elif truelabels[i]==0 and predlabels[i]!=0:
            cm[0][1]=cm[0][1]+1
            
        elif truelabels[i]!=0 and predlabels[i]!=0:
            cm[1][1]=cm[1][1]+1
        
        
    #print(cm)
    return cm 
    
