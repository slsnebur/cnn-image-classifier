import numpy as np
import cv2
import os
from random import shuffle
import progressbar

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

#Python 3.75 maksymalnie
#pip install numpy
#pip install opencv-python
#pip install progressbar2


#TODO sieć konwolucyjna (convnet2d)
#TODO funkcje aktywacji - relu (dla hidden layers)
#TODO Tensorflow + Keras
#TODO test data - randomowe 6250 obrazkow z TEST_DIRECTORY (train/test 80/20)
#TODO nie zmieniaj sciezek ani zadnych nazw przed commitem a jak cos duzego sie bedzie generowac w folderach to dodaj do .gitignore

TRAIN_DIRECTORY = "../../datasets/train"
TEST_DIRECTORY = "../../datasets/test"
IMG_SCALE = 50
TRAIN_ARRAY_FILENAME = "cnn-dog-cat-sc{}.npy".format(IMG_SCALE)
MODEL = 'cnn-img-dogs-cats.model'
LR = 0.001


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
        print("Loading and preprocessing training data IMG_SCALE=" + str(IMG_SCALE) + "\n")

        for img in progressbar.progressbar(os.listdir(TRAIN_DIRECTORY)):
            img_type = identify_train_img(img, "dog", "cat")
            path = os.path.join(TRAIN_DIRECTORY, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
            training_array.append([np.array(img), np.array(img_type)])
        
        shuffle(training_array)
        np.save(array_filename, training_array)

    return training_array

def process_test_data():
    testing_data = []
    for img in progressbar.progressbar(os.listdir(TEST_DIRECTORY)):
        path = os.path.join(TEST_DIRECTORY,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SCALE,IMG_SCALE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



train_data = get_train_data(TRAIN_ARRAY_FILENAME)
test_data = process_test_data()

train = train_data[:-500]
test = train_data[-500:]

#input data
X = np.array([i[0] for i in train]).reshape(-1, IMG_SCALE, IMG_SCALE, 1)
#target data
Y = [i[1] for i in test]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

#model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SCALE, IMG_SCALE, 1)),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

#TODO: nie wiem
model.fit(X, Y, batch_size=10, epochs=5)

