#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-5.

from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import time
import argparse
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
from sklearn.metrics import mean_squared_error

from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

CONFIG_FILE = "config.yml"

TRAIN_DATA_FILE = ""
TEST_DATA_FILE = ""
MODEL_FILE = ''
TAG_POSITIVE = "anormal"

IS_SHUFFLE = False
IS_DROPIN = False

TIMESTEPS = 100  # sub sequence length
DROPIN_COUNT = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
EPOCHS = 60
BATCH_SIZE = 50
TRAIN_START = 0
TRAIN_END = 3000

# File Format
SEP = '\t'
HEADER = None
INDEX_COL = None
USECOLS = None
NROWS = None

# fix random seed for reproducibility
np.random.seed(1234)


def set_gpu(gpu_id):
    """Set the whole GPU environment

    :param gpu_id: list.
    :return:
    """
    if type(gpu_id) == list or gpu_id == None:
        if gpu_id == None:
            gpu_id = ''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)[1:-1]
    else:
        raise TypeError("gpu_id should be a list")


class DataProcessor(object):
    """Data Processing
    """

    def __init__(self, train_data_file, test_data_file):
        """
        Init a DataProcessor
        :param data_file: [timestamp, v0, v1 ... , tag]
        """
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
        """
        Internal Data Initialization
        :return:
        """

        # Raw Dataset. contain tag
        # data = pd.read_table(self.train_data_file, header=None, index_col=0, usecols=[0, 1], nrows=5000)
        # self.data = data.stack()
        # self.data_size = len(self.data)
        # self.train_size = int(self.data_size * 0.8)
        # self.test_size = self.data_size - self.train_size

        # train data
        print("\nCreating train data...\n")
        self.train_data = pd.read_table(self.train_data_file, sep=SEP, header=HEADER, index_col=INDEX_COL,
                                        usecols=USECOLS).stack()
        self.train_size = len(self.train_data)

        train = []
        for index in range(TRAIN_START, TRAIN_END - TIMESTEPS + 1):
            train.append(self.train_data[index: index + TIMESTEPS])
        train = np.array(train).astype(np.float64)  # some dataset is int64 (like power_data)
        train = DataProcessor.normalize(train)

        if IS_SHUFFLE:
            np.random.shuffle(train)

        self.train_x = train[:, :-1]
        self.train_y = train[:, -1]

        if IS_DROPIN:
            self.train_x, self.train_y = DataProcessor.dropin(self.train_x, self.train_y)

        # test data
        print("\nCreating test data...\n")
        self.test_data = pd.read_table(self.test_data_file, sep=SEP, header=HEADER, index_col=INDEX_COL,
                                       usecols=USECOLS).stack()
        self.test_size = len(self.test_data)

        test = []
        for index in range(self.test_size - TIMESTEPS + 1):
            test.append(self.test_data[index: index + TIMESTEPS])
        test = np.array(test).astype(np.float64)
        test = DataProcessor.normalize(test)

        self.test_x = test[:, :-1]
        self.test_y = test[:, -1]

        # Reshape train_x and test_x to LSTM input shape
        self.train_x = self.train_x.reshape((self.train_x.shape[0], 1, self.train_x.shape[1]))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], 1, self.test_x.shape[1]))

        print("\nTrainX Shape: ", self.train_x.shape)
        print("TrainY Shape: ", self.train_y.shape)
        print("TestX Shape: ", self.test_x.shape)
        print("TestY Shape: ", self.test_y.shape)

    @staticmethod
    def normalize(data):
        """
        Normalize the dataset to the same scale
        :param data: array
        :return: normalized array
        """
        data_mean = data.mean()
        data_std = data.std()
        data -= data_mean
        data /= data_std
        return data

    @staticmethod
    def dropin(x, y):
        """ Data Augmentation, the inverse of dropout, i.e. adding more samples.
        :param x: Each row is a training sequence
        :param y: Tne target we train and will later predict
        :return: new augmented X, y
        """
        print(("x shape:", x.shape))
        print(("y shape:", y.shape))
        x_hat = []
        y_hat = []
        for i in range(0, len(x)):
            for j in range(0, np.random.random_integers(0, DROPIN_COUNT)):
                x_hat.append(x[i, :])
                y_hat.append(y[i])
        return np.asarray(x_hat), np.asarray(y_hat)


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
        self.__model = Sequential()

        self.__model.add(LSTM(units=64,
                 input_shape=(self.dp.train_x.shape[1], self.dp.train_x.shape[2]),
                 return_sequences=True))
        self.__model.add(Dropout(0.2))

        self.__model.add(LSTM(units=256, return_sequences=True))
        self.__model.add(Dropout(0.2))

        self.__model.add(LSTM(units=100, return_sequences=False))
        self.__model.add(Dropout(0.2))

        self.__model.add(Dense(units=1))
        self.__model.add(Activation("linear"))

    def __compile_model(self):
        print("\nCompiling Model...\n")
        start = time.time()

        self.__model.compile(loss="mse", optimizer="rmsprop")

        print("Compilation Time : ", time.time() - start)

    def __fit_model(self):
        print("\nTraining Model...\n")
        start = time.time()

        filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=1),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
        ]
        self.__model.fit(self.dp.train_x, self.dp.train_y, batch_size=BATCH_SIZE,
                         epochs=EPOCHS, validation_split=0.05, callbacks=callbacks)

        print("Training Time : ", time.time() - start)

    def predict(self, test_x):
        """
        Do Prediction on X_test, return predicted
        :param model:
        :param testX:
        :return:
        """
        try:
            print("\nPredicting...\n")
            start = time.time()
            predicted = self.__model.predict(test_x)
            print("Predicted Shape: ", predicted.shape)
            print("Prediction Time : ", time.time() - start)

            print("Reshaping predicted")
            predicted = np.ravel(predicted)
            # predicted = np.reshape(predicted, (predicted.size,))

            return predicted
        except KeyboardInterrupt:
            print("prediction exception")

    @staticmethod
    def mse(observation, prediction):
        """Return the MSE array of two ndarray

        :param observation: ndarray
        :param prediction: ndarray
        :return:
        """
        return ((observation - prediction) ** 2) / len(observation)

    @staticmethod
    def rmse(observation, prediction):
        """Return the RMSE array of two ndarray

        :param observation: ndarray
        :param prediction: ndarray
        :return:
        """
        return np.sqrt(((observation - prediction) ** 2) / len(observation))

    @staticmethod
    def threshold(error):
        """
        Define the threshold to get anormal point.
        We can adjust this to get better performance
        :param error: the |y_test - y_hat|
        :return:
        """
        return error.mean()

    @staticmethod
    def plot(testY, prediction):
        """
        Plot the result
        :param testY:
        :param prediction:
        :return:
        """
        try:
            plt.figure(1)

            ax_observation = plt.subplot(311)
            plt.title("Observation")
            plt.plot(testY, 'b')

            plt.subplot(312)
            plt.title("Prediction")
            plt.plot(prediction, 'g')

            plt.subplot(313)
            plt.title("Prediction Error")
            rmse = ModelProcessor.rmse(testY, prediction)
            plt.plot(rmse, 'r')

            x = range(len(testY))
            y = [ModelProcessor.threshold(rmse)] * len(testY)
            plt.plot(x, y, 'r--', lw=4)

            plt.show()
        except Exception as e:
            print("plotting exception")
            print(str(e))

    @staticmethod
    def evaluate(dataset, testY, prediction):
        """
        Do Evaluation. return (precision, recall, f1)
        :param dataset:
        :param testY:
        :param prediction:
        :return:
        """
        testY_data = testY[:, 0].astype(np.float64)
        rmse = ModelProcessor.rmse(testY_data, prediction)
        # retrived_data = dataset.ix[dataset.index[:len(mse)]][mse > ModelProcessor.threshold(mse)]
        retrived_data = testY[rmse > ModelProcessor.threshold(rmse)]
        tpfp = len(retrived_data)
        print("\n[Retrived Data Size] = ", tpfp)

        # retrived_anormal_data = retrived_data[retrived_data['tag'] == TAG_POSITIVE]
        retrived_anormal_data = retrived_data[retrived_data[:, 1] == TAG_POSITIVE]
        tp = len(retrived_anormal_data)
        print("\n[Retrived Anormal Size] = ", tp)

        # real_anormal_data = dataset[dataset['tag'] == TAG_POSITIVE]
        real_anormal_data = testY[testY[:, 1] == TAG_POSITIVE]
        tpfn = len(real_anormal_data)
        print("\n[Real Anormal Size] = ", tpfn)

        precision = tp / tpfp
        recall = tp / tpfn
        f1 = (2 * precision * recall) / (precision + recall) if tp != 0 else 0
        print("\n[Precision] = ", precision)
        print("\n[Recall] = ", recall)
        print("\n[F1] = ", f1)

        return precision, recall, f1


def main(args):
    """
    python lstm_model.py [dynamic]
    :param args: dynamic, will do experiments on dynamic data
    :return:
    """

    global TRAIN_DATA_FILE
    global TEST_DATA_FILE
    global MODEL_FILE
    global IS_SHUFFLE
    global IS_DROPIN
    global TIMESTEPS
    global EPOCHS
    global BATCH_SIZE
    global TRAIN_START
    global TRAIN_END
    global SEP
    global HEADER
    global INDEX_COL
    global USECOLS
    global NROWS

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', type=int, nargs='*', default=None, help='gpu id seprated by blank')
    parser.add_argument('-dtr', '--train-data', dest='train', default=TRAIN_DATA_FILE, help='train data file')
    parser.add_argument('-dte', '--test-data', dest='test', default=TEST_DATA_FILE, help='test data file')
    parser.add_argument('-m', '--model', default=MODEL_FILE, help='model file')
    parser.add_argument('-e', '--epochs', type=int, nargs=1, default=EPOCHS)
    parser.add_argument('-b', '--batch-size', type=int, nargs=1, dest='batchs', default=BATCH_SIZE)
    parser.add_argument('-s', '--shuffle', action='store_true', default=IS_SHUFFLE)
    parser.add_argument('-di', '--dropin', action='store_true', default=IS_DROPIN)
    args = parser.parse_args()

    TRAIN_DATA_FILE = args.train
    TEST_DATA_FILE = args.test
    MODEL_FILE = args.model
    gpu_id = args.gpu
    IS_SHUFFLE = args.shuffle
    IS_DROPIN = args.dropin
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchs

    # Parse Config YAML File
    config = yaml.load(open(CONFIG_FILE))
    datas = config.get('data')
    models = config.get('model')
    dataformat = config.get('dataformat')

    # datas
    TRAIN_DATA_FILE = datas.get('trainFile')
    TEST_DATA_FILE = datas.get('testFile')
    IS_SHUFFLE = datas.get('isShuffle')
    IS_DROPIN = datas.get('isDropin')
    TRAIN_START = datas.get('trainStart')
    TRAIN_END = datas.get('trainEnd')

    # models
    MODEL_FILE = models.get('modelFile')
    EPOCHS = models.get('epochs')
    BATCH_SIZE = models.get('batchSize')
    TIMESTEPS = models.get('timeSteps')

    # Data Format
    SEP = dataformat.get('sep')
    HEADER = dataformat.get('header')
    INDEX_COL = dataformat.get('indexCol')
    USECOLS = dataformat.get('usecols')
    NROWS = dataformat.get('nrows')

    # Set GPU Environment
    set_gpu(gpu_id)

    # Load Data
    print('Creating train and test data... ')
    dp = DataProcessor(TRAIN_DATA_FILE, TEST_DATA_FILE)
    mp = ModelProcessor(dp)

    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        mp.set_model(model)
    else:
        model = mp.get_model()
        model.save(MODEL_FILE)

    # predict
    predicted = mp.predict(dp.test_x)

    # evaluate
    # ModelProcessor.evaluate(dp.dataset, dp.test_y, predicted)

    # plot
    ModelProcessor.plot(dp.test_y, predicted)


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("\nTotal Time Usage: " + str(elapsed))
