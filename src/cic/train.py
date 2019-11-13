import numpy as np
import cv2
import os
import random
from random import shuffle
import progressbar
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

TRAIN_DIRECTORY = "../../datasets/train"
TEST_DIRECTORY = "../../datasets/test"
IMG_SCALE = 50
TRAIN_ARRAY_FILENAME = "dog-cat-sc{}-train.npy".format(IMG_SCALE)
TEST_ARRAY_FILENAME = "dog-cat-sc{}-test.npy".format(IMG_SCALE)
MODEL_NUM = 1
MODEL_NAME = "cnn-{}-img-dogs-cats.{}.model".format(MODEL_NUM, IMG_SCALE)

def identify_train_img(img, first_type, second_type):
    if first_type in img:
        return 1
    else:
        return 0

def get_flip_code(vflip, hflip):
    if hflip or vflip:
        flip_code = -1
    else:
        flip_code = 0 if vflip else 1
    return flip_code

def get_train_data(array_filename):
    training_array = []

    if os.path.exists("./" + array_filename):
        print("Training data already loaded, using: " + array_filename)

        training_array = np.load(array_filename, None, True)
    else:
        print("\nLoading and preprocessing training data IMG_SCALE=" + str(IMG_SCALE))

        for img in progressbar.progressbar(os.listdir(TRAIN_DIRECTORY)):
            img_type = identify_train_img(img, "dog", "cat")
            path = os.path.join(TRAIN_DIRECTORY, img)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SCALE, IMG_SCALE))
            training_array.append([np.array(img), img_type])

            width = img.shape[1]
            height = img.shape[0]
            scale = 1.0

            for i in range(3):
                vertical_flip = random.choice([True, False])
                horizontal_flip = random.choice([True, False])
                #angle = random.random(0, 360) - float v2
                #angle.randrange(0, 360) - int
                angle = random.uniform(0, 360)

                rotate_matrix = cv2.getRotationMatrix2D((IMG_SCALE/2, IMG_SCALE/2), angle, scale)
                flip_code = get_flip_code(vertical_flip, horizontal_flip)

                #flip
                imga = cv2.flip(img, flip_code)
                #rotate
                imga = cv2.warpAffine(img, rotate_matrix, (IMG_SCALE, IMG_SCALE))

                '''
                print(imga.shape)
                plt.imshow(imga, cmap="gray")
                plt.show()
                '''

                training_array.append([np.array(imga), img_type])

        shuffle(training_array)
        np.save(array_filename, training_array)
    print(len(training_array))
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

def get_model(model_num):

    if model_num == 1:
    # 1st model - pretty basic 1 convolutional layer
        model = Sequential()
        # Input layer - 64 filters, kernel size (3x3) with relu activation function
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Output layer
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # 2nd model - 2 Convolutional layers
    elif model_num == 2:
        model = Sequential()
        # Input layer - 64 filters, kernel size (3x3) with relu activation function
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second layer same as first one
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Output layer
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # 3rd model - 2 Convolutional layers
    elif model_num == 3:
        model = Sequential()
        # Input layer - 64 filters, kernel size (3x3) with relu activation function
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third layer with more filters
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Output layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        exit(-1)

    return model

model = get_model(MODEL_NUM)
hist = model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

#Batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 20

hist = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

model.save("model" + str(MODEL_NUM) + "/" + MODEL_NAME)
print("Model saved as " + MODEL_NAME)

# generating plots of accuracy and loss by epochs
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model" + str(MODEL_NUM) + "/" + "accuracy.png")

# summarize history for loss #TODO val_accuracy + val_loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model" + str(MODEL_NUM) + "/" + "loss.png")
