import os
from PIL import Image

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Dense, Flatten, Dropout, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array, random_shift

from keras import backend as K
K.set_image_dim_ordering('th')

from preprocessor import RESIZE

FRAMES = 2
ACTIONS = 2

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def load_data(img_fp='images/', action_fp='actions.csv'):

    y = pd.read_csv(action_fp, header=None).get_values()[:, 1]

    # Initialize img array dataset
    X = np.zeros((y.shape[0], RESIZE[1], RESIZE[0]))

    # Loop through the files
    for img_name in os.listdir(img_fp):
        ind = int(img_name[:-4])
        img = Image.open(img_fp + img_name)
        img_arr = img_to_array(img)[0]
        # Add them to the matrix
        X[ind] = img_arr

        return X[y != -1], y[y != -1]

def normalize(y, frames=FRAMES):
    inds = np.where(y==1)
    percent = np.sum(y[frames-1:]) / y[frames-1:].shape[0]
    amount = (1 - percent) / percent
    return np.hstack([np.arange(frames-1, y.shape[0])] + [inds[0] for i in xrange(int(amount))])

def batch_gen(X, y, frames=FRAMES, batch_size=64, shuffle=True, shifts=False, difference=True):
    # adapted from chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    counter = 0
    sample_index = normalize(y[:, 1])
    number_of_batches = np.ceil(sample_index.shape[0] / batch_size)
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        if difference:
            X_batch = X[batch_index] - X[batch_index-1]
            X_batch = X_batch.reshape(-1, 1, RESIZE[1], RESIZE[0])
        else:
            X_batch = np.zeros((batch_index.shape[0], frames, RESIZE[1], RESIZE[0]))
            for i, ind in enumerate(batch_index):
                if shifts:
                    X_batch[i] = random_shift(X[ind-frames+1:ind+1, :], .1, .1)
                else:
                    X_batch[i] = X[ind - frames + 1:ind + 1, :]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches - 1:
            batch_index = sample_index[batch_size * counter:]
            if difference:
                X_batch = X[batch_index] - X[batch_index - 1]
                X_batch = X_batch.reshape(-1, 1, RESIZE[1], RESIZE[0])
            else:
                X_batch = np.zeros((batch_index.shape[0], frames, RESIZE[1], RESIZE[0]))
                for i, ind in enumerate(batch_index):
                    if shifts:
                        X_batch[i] = random_shift(X[ind - frames + 1:ind + 1, :], .1, .1)
                    else:
                        X_batch[i] = X[ind - frames + 1:ind + 1, :]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def conv_model(frames=FRAMES):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(2, 2),
                            input_shape=(frames, RESIZE[1], RESIZE[0])))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 4, 4))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 4, 4))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 4, 4))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def deep_model():
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(1, RESIZE[1], RESIZE[0])))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 2, 2, subsample=(2, 2), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 2, 2, subsample=(2, 2), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Convolution2D(256, 2, 2, subsample=(2,2), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Convolution2D(512, 2, 2, subsample=(2,2), activation='relu'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def split_data(X, y, split=.9):

    ind = int(y.shape[0] * split)
    return X[:ind], y[:ind], X[ind:], y[ind:]

# Get the data and split it
X_full, y_full = load_data()
# Reformat data for Keras
y_full = to_categorical(y_full, ACTIONS)
Xtr, ytr, Xval, yval = split_data(X_full, y_full)

model = deep_model()
history = model.fit_generator(batch_gen(Xtr, ytr, shifts=False), samples_per_epoch=normalize(ytr[:, 1]).shape[0], nb_epoch=100,
                              validation_data=batch_gen(Xval, yval, shifts=False), nb_val_samples=normalize(yval[:, 1]).shape[0],
                              callbacks=[ModelCheckpoint('conv_model.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                         monitor='val_loss', verbose=0, save_best_only=False,
                                                         save_weights_only=True, mode='auto')])
#
# plt.plot(history.history['val_loss'],'o-')
# plt.plot(history.history['loss'],'o-')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Categorical Crossentropy')
# plt.title('Train Error vs Number of Iterations')
#
# plt.show()