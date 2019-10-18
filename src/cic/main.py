import numpy as np
import cv2
import os
from random import shuffle
import progressbar

TRAIN_DIRECTORY = "../../datasets/train"
TEST_DIRECTORY = "../../datasets/test"
TRAIN_ARRAY_FILENAME = "cnn-dog-cat.npy"
#TODO better serialization
MODEL = 'cnn-img-dogs-cats.model'
IMG_SCALE = 75

def identify_train_img(img, first_type, second_type):
    if first_type in img:
        return [1,0]
    elif second_type in img:
        return [0,1]
    else:
        return [0,0]

def get_train_data(array_filename):
    training_array = []
    if os.path.exists("./" + array_filename):
        print("Training data already loaded, using: " + array_filename)
        
        training_array = np.load(array_filename, None, True)
    else:
        print("Loading and preprocessing training data:\n")

        for img in progressbar.progressbar(os.listdir(TRAIN_DIRECTORY)):
            img_type = identify_train_img(img, "dog", "cat")
            path = os.path.join(TRAIN_DIRECTORY, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
            training_array.append([np.array(img), np.array(img_type)])
        
        shuffle(training_array)
        np.save(array_filename, training_array)

    return training_array

get_train_data(TRAIN_ARRAY_FILENAME)

#TODO sieć konwolucyjna (convnet2d)
#TODO funkcje aktywacji - relu (dla hidden layers)
#TODO Tensorflow + Keras
#TODO test data - randomowe 6250 obrazkow z TEST_DIRECTORY (train/test 80/20)
#TODO nie zmieniaj sciezek ani zadnych nazw przed commitem a jak cos duzego sie bedzie generowac w folderach to dodaj do .gitignore
