#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-9-13


'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function

import sys, os
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import plot_model

import matplotlib.pyplot as plt

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32


def main(args):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    if os.path.exists('blstm.h5'):
        model = load_model('blstm.h5')
    else:
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=4,
                  validation_data=[x_test, y_test], verbose=2)

        model.save('blstm.h5')

    # plot_model(model)
    pred_y = model.predict(x_test)

    plt.figure()
    plt.plot(y_test, 'g')
    plt.plot(pred_y, 'r--')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
