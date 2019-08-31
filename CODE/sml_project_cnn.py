# -*- coding: utf-8 -*-
"""SML_Project_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SAdBkogIX5WsvwAHxtlBC_njOQGo96x8
"""

from google.colab import drive
drive.mount('/content/gdrive')

cd gdrive/My Drive/Colab

ls

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2 as cv
import time
import copy as cp
import pandas as pd
import pickle 

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load1(path_train,path_test1,path_test2):
    transform = transforms.Compose([transforms.Resize(227),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size=32
    
    trainloader = torch.utils.data.DataLoader(
          ImageFilelist(root=path_train,lbl='Training', 
          transform=transform),
          batch_size=batch_size, shuffle=False, pin_memory=True)
    
    testloader1 = torch.utils.data.DataLoader(
          ImageFilelist(root=path_test1,lbl='PublicTest',
          transform=transform),
          batch_size=batch_size, shuffle=False, pin_memory=True)

    testloader2 = torch.utils.data.DataLoader(
          ImageFilelist(root=path_test2,lbl='PrivateTest',
          transform=transform),
          batch_size=batch_size, shuffle=False, pin_memory=True)

          
    return trainloader,testloader1,testloader2

def train(trainloader,net):
    trn=[]; lbl=[]; c=0; t=time.time()
    with torch.no_grad():
         for data1 in trainloader:
             c+=1
             if(c%100==0):
                 print('Training: ',c,', Time: ',time.time()-t)
                 t=time.time()
             images, labels = data1
             images, labels = torch.tensor(images), torch.tensor(labels)
             images, labels = images.to(device), labels.to(device)
             out = net(images)
             for outputs in out:
                trn.append(cp.deepcopy(np.array(outputs.cpu(),np.double)))
             for label in labels:
                lbl.append(cp.deepcopy(np.array(label.cpu(),np.int32)))
    return np.array(trn),np.array(lbl)
         
    
def test(testloader,net):
    tst=[]; lbl=[]; c=0; t=time.time()
    with torch.no_grad():
         for data1 in testloader:
             c+=1
             if(c%50==0):
                 print('Testing: ',c,', Time: ',time.time()-t)
                 t=time.time()
             images, labels = data1
             images, labels = torch.tensor(images), torch.tensor(labels)
             images, labels = images.to(device), labels.to(device)
             out = net(images)
             for outputs in out:  
                tst.append(cp.deepcopy(np.array(outputs.cpu(),np.double)))
             for label in labels:
               lbl.append(cp.deepcopy(np.array(label.cpu(),np.int32)))
    return np.array(tst),np.array(lbl)

def default_flist_reader(flist,lbl):
    imlist=[]
    path='fer2013.csv'
    data=pd.read_csv(path)
    emotion={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad', 5:'Surprise', 6:'Neutral'}
    for i in range(len(data.pixels)):
        if(lbl==data.Usage[i]):
          im=np.array(data.pixels[i].split(' '),np.uint8)
          im.shape=(48,48)
          im=Image.fromarray(im).convert('RGB')
          nm=data.emotion[i]
          imlist.append(cp.deepcopy((im,nm)))    
    print(len(imlist))
    return imlist
   
class ImageFilelist(data.Dataset):
     def __init__(self, root, transform=None,lbl='Training', target_transform=None,
         flist_reader=default_flist_reader):
         self.lbl=lbl
         self.imlist = flist_reader(root,lbl)
         self.transform = transform
         self.target_transform = target_transform
         
   
     def __getitem__(self, index):
         img, target = self.imlist[index]
         #img = self.loader(impath)
         if self.transform is not None:
            img = self.transform(img)
         if self.target_transform is not None:
            target = self.target_transform(target)
         return img, target
   
     def __len__(self):
         return len(self.imlist)

"""**Load Data**"""

p_trn='/Training/'
p_tst1='/PublicTest/'
p_tst2='/PrivateTest/'
trainloader,testloader1,testloader2=load1(p_trn,p_tst1,p_tst2)

alexnet = torchvision.models.alexnet(True)
#vgg = torchvision.models.vgg16(True)
#net=torchvision.models.densenet161(True)
#alexnet.to(device)
#vgg.to(device)
print(alexnet)

net1.classifier=nn.Sequential(*list(alexnet.classifier.children())[:-5])
net2.classifier=nn.Sequential(*list(vgg.classifier.children())[:-3])

"""**Extract** **Features**"""

Xtr1,Ytr1=train(trainloader,net1)
print(Xtr1.shape,Ytr1.shape)
Xtr2,Ytr2=train(trainloader,net2)
print(Xtr2.shape,Ytr2.shape)

Xte1_pub,Yte1_pub=test(testloader1,net1)
print(Xte1_pub.shape,Yte1_pub.shape)

Xte2_pub,Yte2_pub=test(testloader1,net2)
print(Xte2_pub.shape,Yte2_pub.shape)

Xte1_pri,Yte1_pri=test(testloader2,net1)
print(Xte1_pri.shape,Yte1_pri.shape)

Xte2_pri,Yte2_pri=test(testloader2,net2)
print(Xte2_pri.shape,Yte2_pri.shape)

Xtr=[];
for i in range(Ytr1.shape[0]):
  tmp=np.concatenate((Xtr1[i],Xtr2[i]))
  Xtr.append(cp.deepcopy(tmp))
Xtr=np.array(Xtr)
print(Xtr.shape)

Xte_pub=[];
for i in range(Yte1_pub.shape[0]):
  tmp=np.concatenate((Xte1_pub[i],Xte2_pub[i]))
  Xte_pub.append(cp.deepcopy(tmp))
Xte_pub=np.array(Xte_pub)
print(Xte_pub.shape)
  
Xte_pri=[];
for i in range(Yte1_pri.shape[0]):
  tmp=np.concatenate((Xte1_pri[i],Xte2_pri[i]))
  Xte_pri.append(cp.deepcopy(tmp))
Xte_pri=np.array(Xte_pri)
print(Xte_pri.shape)

np.unique(Ytr1)

"""**Preprocessing**"""

Xtr1=preprocessing.scale(Xtr)
Xte1_pub=preprocessing.scale(Xte_pub)
Xte1_pri=preprocessing.scale(Xte_pri)

pca=PCA(n_components=1024)
pca.fit(Xtr)
print(pca.explained_variance_ratio_.sum())

Xtr1=pca.transform(Xtr)
Xte1_pub=pca.transform(Xte_pub)
Xte1_pri=pca.transform(Xte_pri)
print(Xtr1.shape,Xte1_pub.shape,Xte1_pri.shape)

"""**Classification**"""

#clf=LinearSVC()
clf=SVC(kernel='rbf',probability=True)
clf.fit(Xtr1,Ytr1)

"""**------------------------------------------------------------------ Public Data ----------------------------------------------------------------------------**"""

p_lbl=clf.predict(Xte1_pub)
cmat=confusion_matrix(Yte1_pub,p_lbl)
sns.heatmap(cmat,annot=True,fmt='g')
print(clf.score(Xte1_pub,Yte1_pub))

"""ROC Plot"""

prob=clf.predict_proba(Xte1_pub)
scores=[]
for i in range(7):
    fpr, tpr, thresholds = metrics.roc_curve(Yte1_pub, prob[:,i], pos_label=i)
    plt.plot(fpr,tpr,label='C'+str(i))
    plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')  
plt.title(' Receiver Operating Characteristic Curve of Public Data')

"""**------------------------------------------------------------------ Private Data ----------------------------------------------------------------------------**"""

p_lbl=clf.predict(Xte1_pri)
cmat=confusion_matrix(Yte1_pri,p_lbl)
sns.heatmap(cmat,annot=True,fmt='g')
print(clf.score(Xte1_pri,Yte1_pri))

"""ROC Plot"""

prob=clf.predict_proba(Xte1_pri)
scores=[]
for i in range(7):
    fpr, tpr, thresholds = metrics.roc_curve(Yte1_pri, prob[:,i], pos_label=i)
    plt.plot(fpr,tpr,label='C'+str(i))
    plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')  
plt.title('Receiver Operating Characteristic Curve of Private Data')

"""****

**Save** **Features**
"""

ls

with open('Training_feat.pickle', 'wb') as handle1:
    pickle.dump(Xtr, handle1)

with open('Training_lbl.pickle', 'wb') as handle2:
    pickle.dump(Ytr, handle2)

with open('PublicTest_feat.pickle', 'wb') as handle3:
    pickle.dump(Xte_pub, handle3)

with open('PublicTest_lbl.pickle', 'wb') as handle4:
    pickle.dump(Yte_pub, handle4)

with open('PrivateTest_feat.pickle', 'wb') as handle5:
    pickle.dump(Xte_pri, handle5)

with open('PrivateTest_lbl.pickle', 'wb') as handle6:
    pickle.dump(Yte_pri, handle6)

"""**Misc work**"""

import zipfile

!unzip testData.zip

with ZipFile('testData.zip', 'r') as zipObj:
     zipObj.extractall()

"""**References**"""

[1] https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
[2] https://pytorch.org/docs/stable/torchvision/models.html
[3] https://pytorch.org/docs/stable/data.html
[4] https://github.com/pytorch/vision/issues/81