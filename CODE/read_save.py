import numpy as np
import pandas as pd
import copy as cp
import cv2 as cv
import pickle 
   
if __name__=="__main__":
   path='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\SML_dataset\\FER2013\\fer2013\\fer2013.csv'
   data=pd.read_csv(path)
   path_save='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\SML_dataset\\FER2013\\FER\\'
   emotion={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad', 5:'Surprise', 6:'Neutral'}
   for i in range(len(data.pixels)):
      if(i%1000):
         print(i)
      im=np.array(data.pixels[i].split(' '),np.int)
      im.shape=(48,48) 
      img=np.zeros(((48,48,3)))
      img[:,:,0]=im; img[:,:,1]=im; img[:,:,2]=im
      im=np.array(im,np.uint8)
      #print(type(path_save),type(data.Usage[i]),type(data.emotion[i]) )
      name=path_save+data.Usage[i]+'\\'+emotion[data.emotion[i]]+'\\'+str(i)+'.jpg'
      cv.imwrite(name,im)
   
   '''
   im1=np.array(255*(img[0][0]/np.max(img[0][0])),np.uint8)
   im=cv.resize(im1,(240,320))
   im1=cv.GaussianBlur(im,(5,5),5)
   
   im2=abs(np.subtract(im,im1))
   ims1=np.array(255*(im2/np.max(im2)),np.uint8)
   cv.imshow('Image1',ims1)

   im3=np.add(im,im2)
   
   ims=np.array(255*(im3/np.max(im3)),np.uint8)
   cv.imshow('Image',ims)
   '''

