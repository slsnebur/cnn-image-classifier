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
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

TRAIN_DIRECTORY = "../../datasets/train"
TEST_DIRECTORY = "../../datasets/test"
IMG_SCALE = 50
TRAIN_ARRAY_FILENAME = "dog-cat-sc{}-train.npy".format(IMG_SCALE)
TEST_ARRAY_FILENAME = "dog-cat-sc{}-test.npy".format(IMG_SCALE)
MODEL_NAME = "test-cnn-img-dogs-cats.model.{}".format(IMG_SCALE)
LR = 0.001

def identify_train_img(img, first_type, second_type):
    if first_type in img:
        return 1
    else:
        return 0

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

            #augmentacja danych = z jednego obrazka stworzyc 4 innych roznych i zapisac na arrray
            #https://www.kaggle.com/hanzh0420/image-augmentation-with-opencv

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
            training_array.append([np.array(img), img_type])
        
        shuffle(training_array)
        np.save(array_filename, training_array)

    return training_array

#Creating X=[] - training array and Y=[] - target array
X=[]
y=[]
def process_train_data(training_data):
    print("Creating training and testing arrays...\n")
    for img in progressbar.progressbar(training_data):
        X.append(img[0])
        y.append(img[1])

train_data = get_train_data(TRAIN_ARRAY_FILENAME)
process_train_data(train_data)

#converting X,y into numpy arrays
X = np.array(X).reshape(-1, IMG_SCALE, IMG_SCALE, 1)
y = np.array(y)

#normalizing data
X = X/255.0

#Niezbadane sciezki zone
model = Sequential()

#First layer - 64 nodes, kernel size (3,3) with relu activation function
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))

#Second layer is essentialy the same
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='softmax'))

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

BATCH_SIZE = 32
EPOCHS = 10

print(y.shape)

model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

#Saving model
model.save(MODEL_NAME)
print("Model saved as " + MODEL_NAME)