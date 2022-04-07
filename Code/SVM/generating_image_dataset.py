#importing all th required libraries
import cv2
import numpy as np
import os
import pickle
import itertools
import threading
import time
import sys
from numpy import savetxt
from numpy import asarray
import pandas as pd
done = False
#here is the animation skript
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        else:
            sys.stdout.write('\rGenerating dataset .... ' + c)
            sys.stdout.flush()
            time.sleep(0.1)

#to handle the image load found glitch
try:
    #loacting and reading images
    main_directory = input("Enter the working directory--\n")
    folder = input("Enter folder name in which raw images are there..\n")
    Dataset_labels = []
    main_dataset_directory = os.path.join(main_directory,folder)
    for lbls in os.listdir(main_dataset_directory):
        Dataset_labels.append(lbls)
    print("Labels in the dataset are - \n")
    for Dataset_lbls in Dataset_labels:
        print(Dataset_lbls+"\n")

    t = threading.Thread(target=animate)
    t.start()

    output_dataset = []
    output_dataset.append(Dataset_labels)

    #logic to itterate through the images in the directory
    for lbl in Dataset_labels:
        current_path = os.path.join(main_dataset_directory,lbl)
        label  = Dataset_labels.index(lbl)
        for image in os.listdir(current_path):
            image_path = os.path.join(current_path,image)
            wildcats_img = cv2.imread(image_path,0)
            try :
                wildcats_img = cv2.resize(wildcats_img,(300,300))
                img_array = np.array(wildcats_img).flatten()
                output_dataset.append([img_array,label])
            except:
                pass

    done = True
    print("Total images processed -- " + str(len(output_dataset))+"\n")

    #logic to save generated dataset file.
    main_directory=os.path.join(main_directory,"data")
    try:
        os.mkdir(main_directory)
    except:
        pass
    file_name = input("Enter file name of your choice - \n")
    pick_in = open(main_directory + "\\" + file_name + ".pickle",'wb')
    pickle.dump(output_dataset,pick_in)
    pick_in.close()


except:
    print("Wrong Location!!..")
    done = True

input("Press 'Enter' to exit...")
