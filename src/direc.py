from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:

    def __init__(self):
        try:

            model = load_model("model_weights.h5")
        except:
            print("No saved model found. Let's make a new one")
            pass
        else:
            self.model = model
            return
        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self):
        K.set_image_dim_ordering('th')
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to be [samples][pixels][width][height]
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]
        # define baseline model

        # build the model

        # Fit the model
        self.model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=10, batch_size=200, verbose=2)
        # Final evaluation of the model
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        self.model.save("model_weights.h5")

    def predict(self, digit_img):
        import cv2
        # print(digit_img.shape)
        x = digit_img
        x = np.reshape(x, (1, 1, 28, 28)).astype('float32')
        x = x/255
        # print(x)
        res_pr = self.model.predict(x)
        # print(res_pr)
        res = np.argmax(res_pr)

        return res
