#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-9-5.

from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def make_dataset():
    dataset = pd.read_csv('dataset/PRSA_data_2010.1.1-2014.12.31.csv',
                          parse_dates=[['year', 'month', 'day', 'hour']],
                          index_col=0, date_parser=lambda x: datetime.strptime(x, '%Y %m %d %H'))
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv('dataset/pollution.csv')


def plot_dataset():
    # load dataset
    dataset = pd.read_csv('dataset/pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]

    # plot each column
    plt.figure()
    for i in range(len(groups)):
        plt.subplot(len(groups), 1, i + 1)
        plt.plot(values[:, groups[i]])
        plt.title(dataset.columns[groups[i]], y=0.5, loc='right')

    plt.show()


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


TRAIN_DATA_FILE = "dataset/10benchmark/shuttle-unsupervised-ad.tab"
TEST_DATA_FILE = ""

TIMESTEPS = 1
FEATURES = 9

EPOCHS = 60
BATCH_SIZE = 50

# File Format
SEP = '\t'
HEADER = None
INDEX_COL = None
USECOLS = None
NROWS = None


class DataProcessor(object):
    def __init__(self, train_data_file, test_data_file):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.train_data = None
        self.test_data = None
        self.train_size = None
        self.test_size = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.__init_data()

    def __init_data(self):
        print("\nCreating train data...\n")

        self.train_data = pd.read_table(self.train_data_file, sep=SEP,
                                        header=HEADER, index_col=INDEX_COL,
                                        usecols=USECOLS)
        self.train_size = len(self.train_data)

        encoder = LabelEncoder()
        self.train_data.ix[:, FEATURES] = encoder.fit_transform(self.train_data.ix[:, FEATURES])

        normal_data = self.train_data[self.train_data.ix[:, FEATURES] == 0]
        self.train_x = normal_data.ix[:, :FEATURES]
        self.train_y = normal_data.ix[:, FEATURES]

        print("\nCreating test data...\n")
        self.test_data = pd.read_table(self.train_data_file, sep=SEP,
                                        header=HEADER, index_col=INDEX_COL,
                                        usecols=USECOLS)
        self.test_size = len(self.test_data)

        encoder = LabelEncoder()
        self.test_data.ix[:, FEATURES] = encoder.fit_transform(self.test_data.ix[:, FEATURES])


class ModelProcessor(object):
    def __init__(self, dp):
        """Init a ModelProcessor Object

        :param dp: DataProcessor Object
        """
        self.dp = dp
        self.__model = None

    def get_model(self):
        """Get model:
        init model -> compile model -> fit model
        :return:
        """
        if not self.__model:
            self.__init_model()
            self.__compile_model()
            self.__fit_model()

        return self.__model

    def set_model(self, model):
        """Set model
        :param model: from load_model() function
        :return:
        """
        self.__model = model

    def __init_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.dp.train_x.shape[1], self.dp.train_x.shape[2])))
        model.add(Dense(FEATURES))

    def __compile_model(self):
        print("\nCompiling Model...\n")
        start = time.time()

        self.__model.compile(loss='mae', optimizer='adam')

        print("Compilation Time : ", time.time() - start)

    def __fit_model(self):
        print("\nTraining Model...\n")
        start = time.time()

        filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        history = self.__model.fit(self.dp.train_x, self.dp.train_y, batch_size=BATCH_SIZE,
                         epochs=EPOCHS, validation_split=0.05, callbacks=callbacks)

        print("Training Time : ", time.time() - start)


def main(args):
    # load dataset
    dataset = pd.read_csv('dataset/pollution.csv', header=0, index_col=0)
    values = dataset.values

    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])

    # ensure all data is float
    values = values.astype('float32')

    # frame as supervised learning
    reframed = series_to_supervised(values, TIMESTEPS, 1)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    reframed = scaler.fit_transform(reframed)
    reframed = DataFrame(reframed)

    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    # values = reframed.values
    data = reframed.values
    n_train_hours = 365 * 24
    train = data[:n_train_hours, :]
    test = data[n_train_hours:, :]

    # split into input and outputs
    train_x, train_y = train[:, :FEATURES], train[:, FEATURES:]
    test_x, test_y = test[:, :FEATURES], test[:, FEATURES:]

    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], TIMESTEPS, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], TIMESTEPS, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(FEATURES))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=50, batch_size=72,
                        validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)

    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    # make a prediction
    yhat = model.predict(test_x)
    print(yhat.shape)

    # invert scaling for forecast
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    inv_yhat = np.concatenate((yhat, test_x[:, :]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, :FEATURES]

    # invert scaling for observation
    inv_y = np.concatenate((test_y, test_x[:, :]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, :FEATURES]

    # calculate RMSE
    error = np.sqrt((inv_y - inv_yhat) ** 2 / len(inv_y))
    print(error.shape)  # (35039, 8)
    print(values.shape)

    plt.figure()
    for i in range(error.shape[1]):
        plt.subplot(error.shape[1], 1, i+1)
        plt.plot(values[:, i])
        plt.plot(error[:, i], 'r--')
    plt.show()

    # Caculate the accumulated error at each time point
    # plt.plot(error, 'r')
    # plt.show()

    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


if __name__ == "__main__":
    # main(sys.argv)

    df = pd.read_table(TRAIN_DATA_FILE, header=HEADER)
    data_size = df.shape[0]
    features = df.shape[1] -1
    encoder = LabelEncoder()
    df.ix[:, features] = encoder.fit_transform(df.ix[:, features])

    abnormal = df[df.ix[:, features] == 1]
    abnormal_size = abnormal.shape[0]
    normal = df[df.ix[:, features] == 0]
    normal_size = data_size - abnormal_size
    fetch_size = abnormal_size + 100

    plt.figure()
    for col in range(features):
        plt.subplot(features, 1, col + 1)
        plt.plot(df.ix[:fetch_size, col])
        plt.title(col, y=0.5, color='r', loc='right')

        # for row in range(abnormal_size):
        #     if df.ix[row, col] == 1:
        #         plt.plot(row, df.ix[row, col], 'ro')
        #     else:
        #         plt.plot(row, df.ix[row, col], 'g')
    plt.show()

    # plot_dataset()
