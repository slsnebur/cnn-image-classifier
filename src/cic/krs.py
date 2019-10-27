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

TRAIN_DIRECTORY = "../../datasets/train"
TEST_DIRECTORY = "../../datasets/test"
VALID_DIRECTORY = "../../datasets/validation"
IMG_SCALE = 50
TRAIN_ARRAY_FILENAME = "cnn-dog-cat-sc{}.npy".format(IMG_SCALE)
MODEL = 'cnn-img-dogs-cats.model'
LR = 0.001

train_batches = ImageDataGenerator().flow_from_directory(TRAIN_DIRECTORY, target_size=(IMG_SCALE, IMG_SCALE), classes=['dog', 'cat'], batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(TEST_DIRECTORY, target_size=(IMG_SCALE, IMG_SCALE), classes=['dog', 'cat'], batch_size = 10)
valid_batches = ImageDataGenerator().flow_from_directory(TEST_DIRECTORY, target_size=(IMG_SCALE, IMG_SCALE), classes=['dog', 'cat'], batch_size = 10)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SCALE, IMG_SCALE, 3)),
    Flatten(),
    Dense(2, activation='softmax')
])

test_imgs, test_labels = next(test_batches)

model.compile(Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=25, validation_data=valid_batches, validation_steps=4, epochs=50, verbose=2)







