""" Image Classification Project
    This project classifies various wildcats under the five labels namely
    Cheetah
    Jaguar
    Leopard
    Lion
    Tiger

    The project takes an image of the wild cat family applies Support Vector
    Algorithm on it and classifies in into any of the 5 variables.

    Made By - Rahul Dabur
    """

#importing all the necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import random
import itertools
import threading
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        else:
            sys.stdout.write('\rgenerating ' + c)
            sys.stdout.flush()
            time.sleep(0.1)

#main directory reading
main_dir = input( "Enter the path of working directory --\n")
dataset_filename = os.path.join(main_dir,"data")
filename = input("Enter dataset file name --\n")

try:
    #loading and reading dataset
    dataset = open(os.path.join(dataset_filename,filename) + ".pickle","rb")
    t = threading.Thread(target=animate)
    t.start()
    raw_data = pickle.load(dataset)
    dataset.close()
    categories = raw_data[0]
    data = raw_data[1:]
    random.shuffle(data)
    features = []
    labels = []

    #splitting features and labels
    for feature,label in data:
        features.append(feature)
        labels.append(label)

    #test train split
    training_features, testing_features, training_labels, testing_labels = train_test_split(features,labels,test_size=.12)

    #model generation
    model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    model.fit(training_features,training_labels)
    done = True
    print("\nModel Successfully generated...........\n")

    #saing and displaying results
    accuracy = model.score(testing_features,testing_labels)
    print("Accuracy -- " ,accuracy)
    path_model_save = main_dir

    #for data folder navigatio and storing
    path_model_save1=os.path.join(path_model_save,"data")
    try:
        os.mkdir(path_model_save1)
    except:
        pass
    model_save = open(path_model_save1 + "\model.sav","wb")
    pickle.dump(model,model_save)
    model_save.close()

    #for results folder navigation and storage
    path_model_save=os.path.join(path_model_save,"results")
    try:
        os.mkdir(path_model_save)
    except:
        pass

    #logic for confusion matrix display and save
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model, testing_features, testing_labels, display_labels=categories,
            cmap=plt.cm.Blues, normalize=normalize
        )
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
        plt.savefig(os.path.join(path_model_save,title))
    plt.show()

    #for saving prediction results on test data
    predictions = model.predict(testing_features)
    result_array = [["predicted value", "actual_value","right or wrong"]]
    for prediction, actual in list(zip(predictions,testing_labels)):
        if (prediction==actual):
            result_array.append([categories[prediction],categories[actual],"right"])
        else:
            result_array.append([categories[prediction],categories[actual],"wrong"])
    result_array.append(["","",""])
    result_array.append(["","result",""])
    result_array.append(["Accuracy",accuracy,""])
    saving_array = np.array(result_array)
    df = pd.DataFrame(saving_array)
    df.to_csv(os.path.join(path_model_save, 'results.csv'),index=False)

    print("all set")
except:
    print("\nFile not found at the specified directory!!")
    done=True

input("press enter to exit....")
