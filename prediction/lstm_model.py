#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-5.

from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
from sklearn.metrics import mean_squared_error

from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model

TAG_POSITIVE = "anormal"

DATA_FILE = "../dataset/data.csv"
MODEL_FILE = '../models/lstm_model.h5'

IS_SHUFFLE = False
IS_DROPIN = False

TRAIN_SIZE = 6000
EPOCHS = 200
BATCH_SIZE = 50

WINDOW_SIZE = 100  # sub sequence length
DROPIN_COUNT = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function

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

    def __init__(self, data_file):
        """
        Init a DataProcessor
        :param data_file: [timestamp, v0, v1 ... , tag]
        """
        self.data_file = data_file
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None

        self.__init_data()

    def __init_data(self):
        """
        Internal Data Initialization
        :return: 
        """

        # Raw Dataset. contain tag
        self.dataset = pd.read_csv(self.data_file, parse_dates=True, index_col=0)
        # Values of Dataset
        self.data = self.dataset['v0']
        self.data_size = len(self.data)
        self.train_size = int(self.data_size * 0.8)
        self.test_size = self.data_size - self.train_size

        # train data
        print("\nCreating train data...\n")
        train = []
        for index in range(self.train_size - WINDOW_SIZE):
            train.append(self.data[index: index + WINDOW_SIZE])
        train = np.array(train)
        train = DataProcessor.normalize(train)

        if IS_SHUFFLE:
            np.random.shuffle(train)

        self.trainX = train[:, :-1]
        self.trainY = train[:, -1]

        if IS_DROPIN:
            self.trainX, self.trainY = DataProcessor.dropin(self.trainX, self.trainY)

        # test data
        print("\nCreating test data...\n")
        test = []
        tag = []
        for index in range(self.train_size, self.data_size - WINDOW_SIZE):
            test.append(self.data[index: index + WINDOW_SIZE])
            tag.append(self.dataset[index: index + WINDOW_SIZE].ix[-1, -1])
        test = np.array(test)
        test = DataProcessor.normalize(test)
        test = np.c_[test, tag]

        self.testX = test[:, :-2].astype(np.float64)
        self.testY = test[:, -2:]

        # Reshape trainX and testX to LSTM input shape
        self.trainX = self.trainX.reshape((self.trainX.shape[0], self.trainX.shape[1], 1))
        self.testX = self.testX.reshape((self.testX.shape[0], self.testX.shape[1], 1))

        print("\nTrainX Shape: ", self.trainX.shape)
        print("TrainY Shape: ", self.trainY.shape)
        print("TestX Shape: ", self.testX.shape)
        print("TestY Shape: ", self.testY.shape)

    @staticmethod
    def normalize(result):
        """
        Normalize the dataset to the same scale
        :param result: array
        :return: normalized array
        """
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        return result

    @staticmethod
    def dropin(X, y):
        """ Data Augmentation, the inverse of dropout, i.e. adding more samples.
        :param X: Each row is a training sequence
        :param y: Tne target we train and will later predict
        :return: new augmented X, y
        """
        print(("X shape:", X.shape))
        print(("y shape:", y.shape))
        X_hat = []
        y_hat = []
        for i in range(0, len(X)):
            for j in range(0, np.random.random_integers(0, DROPIN_COUNT)):
                X_hat.append(X[i, :])
                y_hat.append(y[i])
        return np.asarray(X_hat), np.asarray(y_hat)


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
        layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

        self.__model.add(LSTM(
            input_length=WINDOW_SIZE - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
        self.__model.add(Dropout(0.2))

        self.__model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
        self.__model.add(Dropout(0.2))

        self.__model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
        self.__model.add(Dropout(0.2))

        self.__model.add(Dense(
            output_dim=layers['output']))
        self.__model.add(Activation("linear"))

    def __compile_model(self):
        print("\nCompiling Model...\n")
        start = time.time()
        self.__model.compile(loss="mse", optimizer="rmsprop")
        print("Compilation Time : ", time.time() - start)

    def __fit_model(self):
        print("\nTraining Model...\n")
        start = time.time()
        self.__model.fit(self.dp.trainX, self.dp.trainY, batch_size=BATCH_SIZE,
                         epochs=EPOCHS, validation_split=0.05)
        print("Training Time : ", time.time() - start)

    def predict(self, testX):
        """
        Do Prediction on X_test, return predicted
        :param model: 
        :param testX: 
        :return: 
        """
        try:
            print("\nPredicting...\n")
            start = time.time()
            predicted = self.__model.predict(testX)
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
        #retrived_data = dataset.ix[dataset.index[:len(mse)]][mse > ModelProcessor.threshold(mse)]
        retrived_data = testY[rmse > ModelProcessor.threshold(rmse)]
        tpfp = len(retrived_data)
        print("\n[Retrived Data Size] = ", tpfp)

        #retrived_anormal_data = retrived_data[retrived_data['tag'] == TAG_POSITIVE]
        retrived_anormal_data = retrived_data[retrived_data[:, 1] == TAG_POSITIVE]
        tp = len(retrived_anormal_data)
        print("\n[Retrived Anormal Size] = ", tp)

        #real_anormal_data = dataset[dataset['tag'] == TAG_POSITIVE]
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

    global DATA_FILE
    global MODEL_FILE
    global IS_SHUFFLE
    global IS_DROPIN
    global EPOCHS
    global BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=DATA_FILE, help='data file')
    parser.add_argument('-m', '--model', default=MODEL_FILE, help='model file')
    parser.add_argument('-gpu', '--gpu', type=int, nargs='*', default=None, help='gpu id seprated by blank')
    parser.add_argument('-e', '--epochs', type=int, nargs=1, default=EPOCHS)
    parser.add_argument('-b', '--batch-size', type=int, nargs=1, dest='batchs', default=BATCH_SIZE)
    parser.add_argument('-s', '--shuffle', action='store_true', default=IS_SHUFFLE)
    parser.add_argument('-di', '--dropin', action='store_true', default=IS_DROPIN)
    args = parser.parse_args()

    DATA_FILE = args.data
    MODEL_FILE = args.model
    gpu_id = args.gpu
    IS_SHUFFLE = args.shuffle
    IS_DROPIN = args.dropin
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchs

    # Set GPU Environment
    set_gpu(gpu_id)

    # Load Data
    print('Creating train and test data... ')
    dp = DataProcessor(DATA_FILE)
    mp = ModelProcessor(dp)

    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        mp.set_model(model)
    else:
        model = mp.get_model()
        model.save(MODEL_FILE)

    # predict
    predicted = mp.predict(dp.testX)

    # evaluate
    ModelProcessor.evaluate(dp.dataset, dp.testY, predicted)

    # plot
    ModelProcessor.plot(dp.testY[:, 0].astype(np.float64), predicted)


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("\nTotal Time Usage: " + str(elapsed))
