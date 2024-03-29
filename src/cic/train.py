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
IMG_SCALE = 70
TRAIN_ARRAY_FILENAME = "dog-cat-sc{}-train.npy".format(IMG_SCALE)
TEST_ARRAY_FILENAME = "dog-cat-sc{}-test.npy".format(IMG_SCALE)
MODEL_NUM = 3
MODEL_NAME = "cnn-{}-img-dogs-cats.{}.model".format(MODEL_NUM, IMG_SCALE)
# Batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 40


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


# Counts number of classes representatives and returns appropriate class weights
def count_test_data(array_data, part_perc_size):
    n_iterations = part_perc_size * len(array_data)
    array_data_t = array_data[:-int(n_iterations)]
    zero_c = 0.
    one_c = 0.

    i = 0
    while i < n_iterations:
        if array_data_t[i] == 0:
            zero_c = zero_c + 1
        else:
            one_c = one_c + 1
        i = i + 1

    # Prints number of classes representatives in train array
    print("Number of 0's in target array = " + str(zero_c))
    print("Number of 1's in target array = " + str(one_c))

    # Calculating class_weights
    l_count = (1. - part_perc_size) * len(array_data)
    zero_t = 0.
    one_t = 0.

    i = 0
    while i < l_count:
        if array_data[i] == 0:
            zero_t = zero_t + 1
        else:
            one_t = one_t + 1
        i = i + 1

    print("Number of 0's in training array = " + str(zero_t))
    print("Number of 1's in training array = " + str(one_t))

    weight_0 = (zero_t + zero_c) / (one_t + one_c + zero_c + zero_t)
    weight_1 = (one_t + one_c) / (one_t + one_c + zero_c + zero_t)
    # normalizing weights
    weight_1 = weight_0/weight_1
    weight_0 = 1.

    matrix = [weight_0, weight_1]
    return matrix



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
                # angle = random.random(0, 360) - float v2
                # angle.randrange(0, 360) - int
                angle = random.uniform(0, 360)
                scale = random.uniform(0.7, 1.3)

                rotate_matrix = cv2.getRotationMatrix2D((IMG_SCALE / 2, IMG_SCALE / 2), angle, scale)
                flip_code = get_flip_code(vertical_flip, horizontal_flip)

                # flip
                imga = cv2.flip(img, flip_code)
                # rotate
                imga = cv2.warpAffine(img, rotate_matrix, (IMG_SCALE, IMG_SCALE))
                
                training_array.append([np.array(imga), img_type])



        shuffle(training_array)
        np.save(array_filename, training_array)
    print(len(training_array))
    return training_array


# Creating X=[] - training array and Y=[] - target array
X = []
y = []


def process_train_data(training_data):
    print("Creating training and testing arrays...\n")
    for img in progressbar.progressbar(training_data):
        X.append(img[0])
        y.append(img[1])


train_data = get_train_data(TRAIN_ARRAY_FILENAME)
process_train_data(train_data)

# converting X,y into numpy arrays
X = np.array(X).reshape(-1, IMG_SCALE, IMG_SCALE, 1)
y = np.array(y)

# Prints number of class representatives and returns correct weights array
weights_array = count_test_data(y, 0.2)
class_weight = {0: weights_array[0], 1: weights_array[1]}

print(class_weight)

# normalizing data
X = X / 255.0


def get_model(model_num):
    if model_num == 1:
        # 1st model - pretty basic 1 convolutional layer
        model = Sequential()
        # Input layer - 64 filters, kernel size (3x3) with relu activation function
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
        # Pooling layer - zredukowanie ilosci cech i zlozonosci obliczeniowej sieci
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattening layer always between convolutional layer and fully connected layer
        # Transforms two dimensional matrix of features into vector and feeds it to FC layer
        model.add(Flatten())
        # Fully connected layer
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
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # 3rd model - 2 Convolutional layers
    elif model_num == 3:
        model = Sequential()
        # Input layer - 32 filters, kernel size (3x3) with relu activation function
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
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
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        exit(-1)

    return model



model = get_model(MODEL_NUM)
hist = model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, class_weight=class_weight)

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