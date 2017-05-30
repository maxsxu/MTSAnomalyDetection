#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-5.

from __future__ import division  # for divide operation in python 2

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

IS_DATA_DYNAMIC = False
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


# TODO
class DataProcessor():
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @staticmethod
    def normalize(result):
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        return result, result_mean

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


# TODO
class ModelProcessor():
    def __index__(self):
        pass


def gen_data():
    """ 
    Generate a synthetic wave by adding up a few sine waves and some noise
    :return: the final wave
    """
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = np.sin(2 * 2 * np.pi * t)
    noise = np.random.normal(0, 0.1, len(t))
    wave1 = wave1 + noise
    print(("wave1", len(wave1)))

    wave2 = np.sin(2 * np.pi * t)
    print(("wave2", len(wave2)))

    t_anormal = np.arange(0.0, 0.5, 0.01)
    wave3 = np.sin(10 * np.pi * t_anormal)
    print(("wave3", len(wave3)))

    tag = [TAG_POSITIVE] * len(t)

    # insert anormal
    insert = int(round(0.8 * len(t)))
    wave1[insert:insert + 50] = wave1[insert:insert + 50] + wave3
    tag[insert:insert + 50] = [TAG_POSITIVE] * len(wave3)

    v0_data = wave1 + wave2

    data = DataFrame([v0_data, tag])
    data = data.T
    data.rename(columns={0: 'v0', 1: 'tag'})

    return data


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
    return result, result_mean


def dropin(X, y):
    """ 
    Data Augmentation, the inverse of dropout, i.e. adding more samples.
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


def create_data(data, train_start, train_end,
                test_start, test_end):
    """
    Split input dataset into train data and test data
    :param data: 
    :param train_start: 
    :param train_end: 
    :param test_start: 
    :param test_end: 
    :return: 
    """
    # train data
    print("\nCreating train data...\n")

    result = []
    for index in range(train_start, train_end - WINDOW_SIZE):
        result.append(data[index: index + WINDOW_SIZE])
    result = np.array(result)  # shape (samples, WINDOW_SIZE)
    result, result_mean = normalize(result)

    print("Mean of train data : ", result_mean)
    print("Shape of train data : ", result.shape)

    train = result[train_start:train_end, :]

    if IS_SHUFFLE:
        np.random.shuffle(train)  # shuffles in-place

    X_train = train[:, :-1]
    y_train = train[:, -1]

    if IS_DROPIN:
        X_train, y_train = dropin(X_train, y_train)

    # test data
    print("\nCreating test data...\n")

    result = []
    for index in range(test_start, test_end - WINDOW_SIZE):
        result.append(data[index: index + WINDOW_SIZE])
    result = np.array(result)  # shape (samples, WINDOW_SIZE)
    result, result_mean = normalize(result)

    print("Mean of test data : ", result_mean)
    print("Shape of test data : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print(("Shape of X_train", np.shape(X_train)))
    print(("Shape of X_test", np.shape(X_test)))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


def create_model(X_train, y_train):
    """
    Comiple and Fit LSTM Model
    :param X_train: 
    :param y_train: 
    :return: 
    """
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
        input_length=WINDOW_SIZE - 1,
        input_dim=layers['input'],
        output_dim=layers['hidden1'],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers['hidden2'],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers['hidden3'],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers['output']))
    model.add(Activation("linear"))

    print("\nCompiling Model...\n")
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)

    print("\nTraining Model...\n")
    start = time.time()
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)
    print("Training Time : ", time.time() - start)

    return model


def predict(model, X_test):
    """
    Do Prediction on X_test, return predicted
    :param model: 
    :param X_test: 
    :return: 
    """
    try:
        print("\nPredicting...\n")
        start = time.time()
        predicted = model.predict(X_test)
        print("Prediction Time : ", time.time() - start)

        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))

        return predicted
    except KeyboardInterrupt:
        print("prediction exception")


def threshold(error):
    """
    Define the threshold to get anormal point.
    We can adjust this to get better performance
    :param error: the |y_test - y_hat|
    :return: 
    """
    return error.mean()


def plot(y_test, predicted):
    """
    Plot the result
    :param y_test: 
    :param predicted: 
    :return: 
    """
    try:
        plt.figure(1)

        ax_observation = plt.subplot(311)
        plt.title("Observation")
        plt.plot(y_test[:len(y_test)], 'b')

        plt.subplot(312)
        plt.title("Prediction")
        plt.plot(predicted[:len(y_test)], 'g')

        plt.subplot(313)
        plt.title("Prediction Error")
        mse = ((y_test - predicted) ** 2) / len(y_test)
        plt.plot(mse, 'r')

        x = range(len(y_test))
        y = [threshold(mse)] * len(y_test)
        plt.plot(x, y, 'r--', lw=4)

        plt.show()
    except Exception as e:
        print("plotting exception")
        print(str(e))


def evaluate(data, y_test, predicted):
    """
    Do Evaluation. return (precision, recall, f1)
    :param data: 
    :param y_test: 
    :param predicted: 
    :return: 
    """
    mse = ((y_test - predicted) ** 2) / len(y_test)
    retrived_data = data.ix[data.index[:len(mse)]][mse > threshold(mse)]
    tpfp = len(retrived_data)
    print("\n[Retrived Data Size] = ", tpfp)

    retrived_anormal_data = retrived_data[retrived_data['tag'] == TAG_POSITIVE]
    tp = len(retrived_anormal_data)
    print("\n[Retrived Anormal Size] = ", tp)

    real_anormal_data = data[data['tag'] == TAG_POSITIVE]
    tpfn = len(real_anormal_data)
    print("\n[Real Anormal Size] = ", tpfn)

    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    print("\n[Precision] = ", precision)
    print("\n[Recall] = ", recall)
    print("\n[F1] = ", f1)

    return precision, recall, f1


def run_network(data=None, model=None):
    if data is None:
        data = gen_data()
        print(("Length of Data", len(data)))

    print('Creating train and test data... ')
    X_train, y_train, X_test, y_test = create_data(data['v0'], 0, TRAIN_SIZE, 0, len(data))

    if model is None:
        model = create_model(X_train, y_train)
        model.save(MODEL_FILE)

    # predict
    predicted = predict(model, X_test)

    # evaluate
    evaluate(data, y_test, predicted)

    # plot
    plot(y_test, predicted)

    return model, y_test, predicted


def main(args):
    """
    python lstm_model.py [dynamic]
    :param args: dynamic, will do experiments on dynamic data
    :return: 
    """

    global DATA_FILE
    global MODEL_FILE
    global IS_DATA_DYNAMIC
    global IS_SHUFFLE
    global IS_DROPIN
    global EPOCHS
    global BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', default=DATA_FILE, help='data file')
    parser.add_argument('-model', '--model', default=MODEL_FILE, help='model file')
    parser.add_argument('-gpu', '--gpu', type=int, nargs='*', default=None, help='gpu id seprated by blank')
    parser.add_argument('-epochs', '--epochs', type=int, nargs=1, default=EPOCHS)
    parser.add_argument('-batchs', '--batch-size', type=int, nargs=1, dest='batchs', default=BATCH_SIZE)
    parser.add_argument('-d', '--dynamic', action='store_true', default=IS_DATA_DYNAMIC)
    parser.add_argument('-shuffle', '--shuffle', action='store_true', default=IS_SHUFFLE)
    parser.add_argument('-dropin', '--dropin', action='store_true', default=IS_DROPIN)
    args = parser.parse_args()

    DATA_FILE = args.data
    MODEL_FILE = args.model
    gpu_id = args.gpu
    IS_DATA_DYNAMIC = args.dynamic
    IS_SHUFFLE = args.shuffle
    IS_DROPIN = args.dropin
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchs

    # Set GPU Environment
    set_gpu(gpu_id)

    if IS_DATA_DYNAMIC:
        data = gen_data()
    else:
        data = pd.read_csv(DATA_FILE, parse_dates=True, index_col=0)

    print(("Data Size = ", len(data)))

    # Load Trained Model
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
    else:
        model = None

    run_network(data=data, model=model)


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("\nTotal Time Usage: " + str(elapsed))
