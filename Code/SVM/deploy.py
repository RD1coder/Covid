# -*- coding: utf-8 -*-
"""
Created on Wed March 2 11:36:37 2022

@author: rahul
"""

import numpy as np
import cv2
import os
import pickle
import random
from sklearn.svm import SVC

#loading model
main_dir = input("Enter the path of the working directory\n")
model_dir = os.path.join(main_dir,"data")
model_var = open(os.path.join(model_dir,"model.sav"),"rb")
model = pickle.load(model_var)
model_var.close()

#geting dataset file name
Dataset_labels = []
for lbls in os.listdir(model_dir):
    Dataset_labels.append(lbls)
for item in Dataset_labels:
    if "pickle" in item:
        file = item
        break

#collecting dataset
dataset = open(os.path.join(model_dir,file),"rb")
raw_data = pickle.load(dataset)
dataset.close()
categories = raw_data[0]
data = raw_data[1:]
random.shuffle(data)
features = []
labels = []

#feature label split
for feature,label in data:
    features.append(feature)
    labels.append(label)

c = 1
while (c == 1):
    img = input("Test image name without format-- \n")
    temp = []
    for lbls in os.listdir(main_dir):
        temp.append(lbls)
    format = [".jpg",".png",".tif",".tiff",".bmp",".jpeg"]
    img_name =""
    for item in temp:
        for f in format:
            if (img + f == item):
                img_name = img + f
                break

    testing_features =[]
    image_path = os.path.join(main_dir,img_name)
    try:
        image = cv2.imread(image_path,0)
        image = cv2.resize(image,(300,300))
        img_array = np.array(image).flatten()
        testing_features.append(img_array)
        testing_features.append(img_array)
        prediction = model.predict(testing_features)
        label = categories[prediction[-1]]
        cv2.imshow(label,testing_features[-1].reshape(300,300))
        print("The image shown corresponds to -- " + label+"\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        x=input("yeah!! your previous pridiction was succesfully!!!\ntype 'y' to make another prediction...\n 'n' to abort... \n")
        x= x.strip(" ")
        if (x=="Y" or x=="y"):
            c=1
        else:
            c=0

    except:
        print("Can't find the file.... please enter correct path....\n")
        abort = input ("type 'x' to abort, or just hot enter to go with the flow...")
        if (abort == "x"):
            c=0
        else:
            pass
