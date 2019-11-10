import numpy as np
import cv2
import os
import sys
from random import shuffle
import progressbar

import keras

#Images to predict and classify
PREDICT_DIR = "../../predict"
#Model
MODEL_NAME = sys.argv[1]
IMG_SCALE = int(MODEL_NAME.split('.')[1])

from keras.models import load_model
model = load_model(MODEL_NAME)

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

def process_images():

    img_array = []
    for img in progressbar.progressbar(os.listdir(PREDICT_DIR)):
        path = os.path.join(PREDICT_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
        img_array.append([np.array(img)])
    
    return img_array

X = []
X = process_images()
X = np.array(X).reshape(-1, IMG_SCALE, IMG_SCALE, 1)
X = X/255.0

#dog = 1
#cat = 0
print(model.predict(X, verbose=1))