# coding=utf-8

# Created by max on 17-12-28

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import time
import argparse
import yaml

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pandas import Series, DataFrame

from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics

from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from matplotlib.patches import Ellipse

# plt.ioff()
# matplotlib.use('Agg')

CONFIG_FILE = "config.yml"

dataset_trainrange = {
    "../dataset/mts/labeled/data1_occupancy_ad_5d.csv": (2900, 6654),
    "../dataset/mts/labeled/data2_occupancy_ad_5d.csv": [(3081, 6830), (6700, 9000)],
    "../dataset/mts/labeled/data3_occupancy_ad_5d.csv": [(2900, 6654), (6000, 7500)],
    "../dataset/mts/labeled/eeg_eye_state_14d.csv": (9057, 11105)
}

TRAIN_DATA_FILE = "../dataset/mts/labeled/data3_occupancy_ad_5d.csv"
TEST_DATA_FILE = "../dataset/mts/labeled/data3_occupancy_ad_5d.csv"
MODEL_FILE = '../result/model/data3.h5'
TAG_POSITIVE = 1

TRAIN_START = dataset_trainrange[TRAIN_DATA_FILE][0][0]
TRAIN_END = dataset_trainrange[TRAIN_DATA_FILE][0][1]
TEST_STATT = dataset_trainrange[TRAIN_DATA_FILE][1][0]
TEST_END = dataset_trainrange[TRAIN_DATA_FILE][1][1]
TIMESTEPS = 100  # sub sequence length
BATCH_SIZE = 50
EPOCHS = 60

IS_SHUFFLE = False
IS_DROPIN = False
DROPIN_COUNT = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function

# File Format
SEP = ','
HEADER = 0
INDEX_COL = 't'
USECOLS = None
NROWS = None

# fix random seed for reproducibility
np.random.seed(1234)


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

        # raw
        self.train_data = None
        self.test_data = None
        self.train_values = None
        self.test_values = None
        self.train_tag = None
        self.test_tag = None
        self.train_size = None
        self.test_size = None
        self.features = None

        # train & test
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.__init_data()

    def __init_data(self):
        """Internal Data Initialization

        """

        # train data
        print("\nCreating train data...\n")
        self.train_data = pd.read_table(self.train_data_file, sep=SEP, header=HEADER, index_col=INDEX_COL,
                                        usecols=USECOLS).iloc[TRAIN_START:TRAIN_END + 1, :]
        self.train_data.iloc[:, :-1] = self.train_data.iloc[:, :-1].astype(np.float64)
        self.train_data.iloc[:, -1] = self.train_data.iloc[:, -1].astype(np.int)
        self.train_values = self.train_data.values[:, :-1]
        self.train_tag = self.train_data.values[:, -1]
        self.train_size = len(self.train_data)
        self.features = self.train_values.shape[1]

        reframed = DataProcessor.series_to_supervised(self.train_values, TIMESTEPS, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        reframed = scaler.fit_transform(reframed)
        reframed = DataFrame(reframed)

        reframed_values = reframed.values
        self.train_x = reframed_values[:, :TIMESTEPS * self.features]
        self.train_y = reframed_values[:, TIMESTEPS * self.features:]

        # test data
        print("\nCreating test data...\n")
        self.test_data = pd.read_table(self.test_data_file, sep=SEP, header=HEADER, index_col=INDEX_COL,
                                       usecols=USECOLS).iloc[TEST_STATT:TEST_END + 1, :]
        self.test_data.iloc[:, :-1] = self.test_data.iloc[:, :-1].astype(np.float64)
        self.test_data.iloc[:, -1] = self.test_data.iloc[:, -1].astype(np.int)
        self.test_values = self.test_data.values[:, :-1]
        self.test_tag = self.test_data.values[:, -1]
        self.test_size = len(self.test_data)

        reframed = DataProcessor.series_to_supervised(self.test_values, TIMESTEPS, 1)
        self.test_tag = Series(self.test_tag).shift(TIMESTEPS).dropna().astype(np.int).values

        scaler = MinMaxScaler(feature_range=(0, 1))
        reframed = scaler.fit_transform(reframed)
        reframed = DataFrame(reframed)

        reframed_values = reframed.values
        self.test_x = reframed_values[:, :TIMESTEPS * self.features]
        self.test_y = reframed_values[:, TIMESTEPS * self.features:]

        # Reshape train_x and test_x to LSTM input shape
        self.train_x = self.train_x.reshape((self.train_x.shape[0], TIMESTEPS, self.features))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], TIMESTEPS, self.features))

        print("\nTrainX Shape: ", self.train_x.shape)
        print("TrainY Shape: ", self.train_y.shape)
        print("TrainTag Shape: ", self.train_tag.shape)
        print("TestData Shape: ", self.test_data.shape)
        print("TestX Shape: ", self.test_x.shape)
        print("TestY Shape: ", self.test_y.shape)
        print("TestTag Shape: ", self.test_tag.shape)
        print("Observed Anomalied Count: ", len(self.test_tag[self.test_tag == TAG_POSITIVE]))

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

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """Convert series to supervised learning.

        Args:
            data:
            n_in:
            n_out:
            dropnan:

        Returns:

        """
        features = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = [], []

        # x sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(features)]

        # y sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(features)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(features)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg


class ModelProcessor(object):
    def __init__(self, dp):
        """Init a ModelProcessor Object

        :param dp: DataProcessor Object
        """
        self.dp = dp
        self.pred_y = None
        self.pred_tag = None
        self.percentile = None
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

        self.__model.add(Dense(units=self.dp.features))
        self.__model.add(Activation("relu"))

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
                         epochs=EPOCHS, validation_split=0.05, callbacks=callbacks, verbose=2)

        print("Training Time : ", time.time() - start)

    def predict(self, test_x):
        """Prediction on X_test, return pred_y

        Args:
            test_x:

        Returns:

        """
        try:
            print("\nPredicting...\n")
            start = time.time()

            self.pred_y = self.__model.predict(test_x)
            print("Predicted Shape: ", self.pred_y.shape)

            print("Prediction Time : ", time.time() - start)

            return self.pred_y
        except KeyboardInterrupt:
            print("prediction exception")

    def evaluate(self, percentile):
        """Evaluation. return (precision, recall, f1)

        Returns:

        """
        errors = ModelProcessor.mae(self.dp.test_y, self.pred_y)
        self.percentile = percentile

        true_tag = self.dp.test_tag
        self.pred_tag = np.full(true_tag.shape, -1)
        self.pred_tag[errors > ModelProcessor.threshold(errors, self.percentile)] = 1

        precision = ModelProcessor.precision_score(true_tag, self.pred_tag)
        recall = ModelProcessor.recall_score(true_tag, self.pred_tag)
        f1 = ModelProcessor.f1_score(true_tag, self.pred_tag)

        return precision, recall, f1

    @staticmethod
    def threshold(errors, q):
        """Define the threshold to get anormal point.
        We can adjust this to get better performance

        Args:
            errors (1-D np.ndarray): the |y_test - y_hat|

        Returns:

        """
        return np.percentile(errors, q)

    @staticmethod
    def mae(true_y, pred_y):
        errors = []
        for i in range(len(true_y)):
            errors.append(np.sum(np.absolute(true_y[i, :] - pred_y[i, :])) / true_y.shape[1])
        errors = np.array(errors)

        return errors

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
    def plot_mts_anomalies(mts,
                           dataset_name,
                           label_obserced_index=-2,
                           label_predicted_index=-1,
                           label_obserced_anomaly=1,
                           label_predicted_anomaly=1,
                           dimension_show=None,
                           **kwargs):
        """Plot Anomalies of MTS.

        Args:
            mts (DataFrame): `MTS DataFrame` or `MTS Result DataFrame` (with predicted labels attached).
            dataset_name (str): show in plot title.
            label_obserced_index (int): label index of observation. `-2` for Default.
            label_predicted_index (int): label index of predication. `-1` for Default.
            label_obserced_anomaly (str or int): 1 or 'a' or other. `1` for Default.
            label_predicted_anomaly (str or int or None): 1 or 'a' or other. `None` for only plot observations. `1` for Default.
            dimension_show (int): control the number of dimensions to show in plot. None for show all dimensions.
            kwargs (dict): precision, recall, f1, threshold, etc. Extra Data to show.

        Examples:
            | t | v0 | v1 | v2 | tag_true | tag_pred |
            | 0 | 0.81 | 0.13 | 0.77 | 1 | 1 |
            | 1 | 0.72 | 0.32 | 0.65 | -1 | 1 |

        """
        if kwargs is not None:
            percentile = kwargs.pop('percentile')
            precision = kwargs.pop('precision')
            recall = kwargs.pop('recall')
            f1 = kwargs.pop('f1')

        # Convert MTS DataFrame Index to sequence of [0, LEN-1]
        mts.reset_index(drop=True, inplace=True)

        # Dimension to show
        if dimension_show is None:
            if label_predicted_index is not None:
                features = mts.shape[1] - 2
            else:
                features = mts.shape[1] - 1
            print("Features=", features)
        else:
            features = dimension_show

        # Observed and Predicated Anomalies
        anomalies_observed = mts.iloc[:, :label_obserced_index][
            mts.iloc[:, label_obserced_index] == label_obserced_anomaly]

        if label_predicted_anomaly is not None:
            anomalies_predicted = mts.iloc[:, :label_obserced_index][
                mts.iloc[:, label_predicted_index] == label_predicted_anomaly]

        # Plot
        fig, axes = plt.subplots(features, 1, sharex='all')
        for i in range(features):
            ax = axes[i]
            ax.plot(mts.iloc[:, i], color='black')
            ax.set_ylabel("{}".format(mts.columns[i]))

            x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
            x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
            dx = x1 - x0
            dy = y1 - y0
            maxd = max(dx, dy)
            width = 2.5 * maxd / dx
            height = 2.5 * maxd / dy

            if i == 0:
                ax.set_title("Anomalies of {}".format(dataset_name[dataset_name.rfind('/') + 1:]))

            if i == features - 1:
                xlabel = "Time"
                xlabel += "\n\nPercentile={:d}  Precision={:.3f}  Recall={:.3f}  F1={:.3f}" \
                    .format(percentile, precision, recall, f1)

                ax.set_xlabel(xlabel)

            # Plot Observed Anomalies
            for x, y in zip(anomalies_observed.index, anomalies_observed.values[:, i]):
                xy = (x, y)
                circle = Ellipse(xy, width, height, fill=False, edgecolor='green')
                ax.add_patch(circle)

            # Plot Predicted Anomalies
            if label_predicted_anomaly is not None:
                for x, y in zip(anomalies_predicted.index, anomalies_predicted.values[:, i]):
                    xy = (x + 0.2, y + 0.2)
                    circle = Ellipse(xy, width, height, fill=False, edgecolor='red')
                    ax.add_patch(circle)

        plt.grid()
        plt.savefig(dataset_name + "_" + str(percentile), format='png')
        # plt.show()


def main(args):
    dp = DataProcessor(TRAIN_DATA_FILE, TEST_DATA_FILE)
    mp = ModelProcessor(dp)

    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        mp.set_model(model)
    else:
        model = mp.get_model()
        model.save(MODEL_FILE)

    # predict
    pred_y = mp.predict(dp.test_x)

    # Threshold Percentile vs. P/R/F1
    result_scores = []
    for q in range(0, 101, 10):
        # evaluate
        precision, recall, f1 = mp.evaluate(q)

        result_scores.append([q, precision, recall, f1])

        # plot
        mts_result = dp.test_data.shift(TIMESTEPS).dropna()
        mts_result.iloc[:, -1] = mts_result.iloc[:, -1].astype(np.int)
        mts_result['tag_pred'] = mp.pred_tag
        mts_result['tag_pred'] = mts_result['tag_pred'].astype(np.int)
        ModelProcessor.plot_mts_anomalies(mts_result, TEST_DATA_FILE.split('/')[-1][:-4],
                                          precision=precision, recall=recall, f1=f1,
                                          percentile=mp.percentile)

    DataFrame(result_scores).set_index(0).to_csv(
        "{}-scores.csv".format(TEST_DATA_FILE.split('/')[-1][:-4]),
        index_label='q',
        header=['p', 'r', 'f1'])


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("Used {0:0.3f} seconds".format(elapsed))
