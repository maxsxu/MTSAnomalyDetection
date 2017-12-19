#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-10-10

from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import time
import random

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager

from pandas import DataFrame

import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import IsolationForest

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.base import BaseEstimator, ClassifierMixin
from keras.wrappers.scikit_learn import KerasClassifier


__all__ = ["DataProcessor", "ModelProcessor"]


TAG_POSITIVE = "o"

TIMESTEPS = 1
TRAIN_PERCENTAGE = 0.8

EPOCHS = 60
BATCH_SIZE = 50


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """convert series to supervised learning
    :param data: array like
    :param n_in:
    :param n_out:
    :param dropnan:
    :return:
    """
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


class BaseDataProcessor(object):
    """ Base Data Processor

    """

    def __init__(self, train_data_file, test_data_file=None,
                 sep='\t', index_col=None):
        """ Init a DataProcessor

        :param train_data_file:
        :param test_data_file:
        :param sep: data file seperator. '\\\\t' for table, ',' for csv
        :param index_col: 'col_name' or col_index
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

        self.sep = sep
        self.index_col = index_col

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
    def shuffle(data):
        return np.random.shuffle(data)

    @staticmethod
    def dropin(x, y, dropin_count=10):
        """ Data Augmentation, the inverse of dropout, i.e. adding more samples.
        :param x: Each row is a training sequence
        :param y: Tne target we train and will later predict
        :param dropin_count: each sample randomly duplicated between 0 and 9 times
        :return: new augmented X, y
        """
        x_hat = []
        y_hat = []
        for i in range(0, len(x)):
            for j in range(0, np.random.random_integers(0, dropin_count)):
                x_hat.append(x[i, :])
                y_hat.append(y[i])
        return np.asarray(x_hat), np.asarray(y_hat)

    @staticmethod
    def precision_score(true_y, pred_y, decimals=3):
        p = metrics.precision_score(true_y, pred_y)
        if decimals:
            return round(p, ndigits=decimals)
        else:
            return p

    @staticmethod
    def recall_score(true_y, pred_y, decimals=3):
        r = metrics.recall_score(true_y, pred_y)
        if decimals:
            return round(r, ndigits=decimals)
        else:
            return r

    @staticmethod
    def f1_score(true_y, pred_y, decimals=3):
        f1 = metrics.f1_score(true_y, pred_y)
        if decimals:
            return round(f1, ndigits=decimals)
        else:
            return f1

    @staticmethod
    def plot_result(file_name, train_x, test_y, pred_y,
                    precision, recall, f1):
        plt.figure(figsize=(16, 9))
        plt.subplot(211)
        plt.plot(train_x)
        plt.title("Observation")

        ax = plt.subplot(212)
        plt.ylim([-0.5, 1.5])
        plt.plot(test_y, 'g', lw=1.5, label="true_y")
        plt.plot(pred_y, 'r--', lw=1.5, label="pred_y")
        plt.text(0, 1, ' Precision=%.2f \n Recall=%.2f \n F1=%.2f' % (precision, recall, f1),
                 ha='left', va='top', fontsize=12, transform=ax.transAxes)
        plt.legend()
        plt.title("Result")

        plt.savefig("result/%s-result.png" % file_name)

    @staticmethod
    def plot_prf(file_name,
                 precision_list,
                 recall_list,
                 f1_list,
                 bar_labels):
        """Plot P-R-F1 Comparision bar
        :param file_name:
        :param precision_list:
        :param recall_list:
        :param f1_list:
        :param bar_labels:
        :return:
        """
        plt.figure(figsize=(16, 9))
        xbar = np.arange(len(precision_list))
        total_width, n = 0.8, 3
        width = total_width / n
        xbar = xbar - (total_width - width) / 2

        plt.bar(xbar, precision_list, width=width, label='Precision')
        plt.bar(xbar + width, recall_list, width=width, label='Recall', tick_label=bar_labels)
        plt.bar(xbar + 2 * width, f1_list, width=width, label='F1')
        plt.legend()
        plt.title(file_name)

        plt.savefig("result/%s-prf.png" % file_name)

    @staticmethod
    def plot_prc(file_name, precisions, recalls, average_precision):
        # Plot P-R curve
        plt.figure(figsize=(16, 9))
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
        plt.savefig("result/%s-pr.png" % file_name)

    @staticmethod
    def log_prf(ds_precision, ds_recall, ds_f1, header):
        """ Save precision, recall, f1 of each model

        :param ds_precision: list. [dataset,[precision of each model]]
        :param ds_recall: list. [dataset,[recall of each model][
        :param ds_f1: list. [dataset,[f1 of each model]]
        :param header: list. [model_names] csv header
        :return:
        """
        file_names = ('p.csv', 'r.csv', 'f1.csv')
        for file_name, ds_score in zip(file_names, (ds_precision, ds_recall, ds_f1)):
            np.savetxt(file_name, ds_precision, delimiter=',', fmt='%s',
                       header=str(header)[1:-1], comments='')


class BaseModelProcessor(object):
    """Base BaseModelProcessor

    """

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
        pass

    def __compile_model(self):
        pass

    def __fit_model(self):
        pass

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


class DataProcessor(BaseDataProcessor):
    """ Supervisord Data Processor (with label)
    anormal: 1
    normal: 0

    1. train_data = anormal*80% + normal*80%
    2. test_data = anormal*20% + normal*20%
    3. Shuffle and Dropin
    """

    def __init__(self, train_data_file, test_data_file=None,
                 sep='\t', index_col=None):
        """ Init a DataProcessor

        :param train_data_file:
        :param test_data_file:
        :param sep: data file seperator. '\\\\t' for table, ',' for csv
        :param index_col: 'col_name' or col_index
        """
        super(DataProcessor, self).__init__(train_data_file, test_data_file, sep, index_col)
        self.__init_data()

    def __init_data(self):
        df = pd.read_table(self.train_data_file, header=None, sep=self.sep, index_col=self.index_col)
        features = df.shape[1] - 1

        # Encode label (y)
        if df.ix[:, features].dtype == 'object':
            encoder = LabelEncoder()
            df.ix[:, features] = encoder.fit_transform(df.ix[:, features])

        data = df.values
        data_size = len(data)

        anormal_data = data[data[:, -1] == 1]
        normal_data = data[data[:, -1] != 1]
        anormal_size = len(anormal_data)
        normal_size = len(normal_data)

        train_anormal_size = int(anormal_size * TRAIN_PERCENTAGE)
        train_normal_size = int(normal_size * TRAIN_PERCENTAGE)

        self.train_data = np.r_[anormal_data[:train_anormal_size, :],
                                normal_data[:train_normal_size, :]]
        self.test_data = np.r_[anormal_data[train_anormal_size:, :],
                               normal_data[train_normal_size:, :]]

        # Shuffle
        DataProcessor.shuffle(self.train_data)
        DataProcessor.shuffle(self.test_data)

        # X and Y
        self.train_x = self.train_data[:, :features]
        self.train_y = self.train_data[:, features]

        self.test_x = self.test_data[:, :features]
        self.test_y = self.test_data[:, features]

        # Dropin
        self.train_x, self.train_y = DataProcessor.dropin(self.train_x, self.train_y)


class ModelProcessor(BaseModelProcessor):
    def __init__(self, dp):
        """Init a ModelProcessor Object

        :param dp: DataProcessor Object
        """
        super(ModelProcessor, self).__init__(dp)

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


if __name__ == "__main__":
    pass
