# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 22:32:23 2018

@author: yonic
"""

import tarfile
import cv2
import numpy as np
from PIL import Image
import os

DB_FOLDER = "/home/yonic/repos/CycleGAN-Tensorflow-PyTorch/db"
FACE_RECOGNIZE_TRAIN_FOLDER = "/home/yonic/repos/deepcharacter/data"

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class Importer:
    def __init__(self,fname):
        self.fname = os.path.join(DB_FOLDER,fname)
        self.tar_db = tarfile.open(self.fname)
        self.catalog = []
        i = self.tar_db.next()
        while i:
            self.catalog.append(i.get_info())
            i = self.tar_db.next()
        self.catalog = sorted(self.catalog,key=lambda x:x['name'])
        lbp_train_data = os.path.join(FACE_RECOGNIZE_TRAIN_FOLDER,'lbpcascade_frontalface.xml')
        
        self.f_cascade = cv2.CascadeClassifier(lbp_train_data)
        
    '''Returns : - If is manages to detect face returns cropped face,
                 - True is gaced is recognized
    '''
    def detect_faces(self, colored_img, scaleFactor = 1.1):
        #just making a copy of image passed, so that passed image is not changed 
        img_copy = colored_img.copy()

        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

        #let's detect multiscale (some images may be closer to camera than others) images
        faces = self.f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

        #go over list of faces and draw them as rectangles on original colored img
        #assert(len(faces) == 1) #it is supposed just one face in this data base
        for (x, y, w, h) in faces:
          #cv2.rectangle(img_copy, (x-50, y-80), (x+w+50, y+h+50), (0, 255, 0), 2)
          x1 = max(0,x - 50)
          y1 = max(0,y-80)
          x2 = min(img_copy.shape[1],x+w+50)
          y2 = min(img_copy.shape[0],y+h+50)
          img_copy = img_copy[y1:y2,x1:x2,:]
      
        if len(faces)>0:
            recognized = True
        else:
             recognized = False
        return img_copy,recognized
    
    
            
    def get_full_catalog(self):
        return self.catalog
    
    def get_sorted_image_name(self):
        return [f['name'] for f in self.catalog][1:]#first is a folder name!
    
    def get_stream(self,tar_fname):
        fileobj = self.tar_db.extractfile(tar_fname)
        return fileobj
    
    '''Return extract RGB image
    '''
    def get_image(self,tar_fname):
        image = Image.open(self.get_stream(tar_fname))
        image = np.array(image)
        return  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
  
def test__(tar_fname="cameron_images.tgz"):
    i = Importer(tar_fname)
    image_names = i.get_sorted_image_name()
    image = i.get_image(image_names[1000])
    face_detected = i.detect_faces(image)
    cv2.imshow('image',face_detected)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()  
    
def test(tar_fname="teresa_images.tgz"):
    i = Importer(tar_fname)
    image_names = i.get_sorted_image_name()
    detected_stat = 0.0
    face_shape = None
    
    for index in range(len(image_names)):
        image = i.get_image(image_names[index])
        face,if_detected = i.detect_faces(image)
        if face_shape is None:
            face_shape = np.array(face.shape)
        else:
            face_shape += np.array(face.shape)
        if if_detected:
            detected_stat += 1.0
        if (index + 1) % 100 == 0:
            print("Step:{}, ratio:{}, shape:{}, orig shape:{}".
                  format(index,detected_stat/index,face_shape/index,image.shape))
            
    print("db:{}, detected ratio:{}, shape:{}".
          format(tar_fname,detected_stat/len(image_names)),face_shape/index)
    
    

if __name__ == "__main__":
    test()
    
        
    