#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-5.

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model

DATA_FILE = "dataset/data0.csv"
MODEL_FILE = 'lstm_model.h5'

DATA_SIZE = 500
WINDOW_SIZE = 10

bath_size = 1
epochs = 1500
units = 1

# Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# Create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# Invert differenced value
def invert_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# Scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler


# Invert scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Fit an LSTM network to training data
def fit_lstm(train, batch_size, epochs, units):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Create Model
    model = Sequential()
    model.add(LSTM(units, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit Model
    for i in range(epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    return model


# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)


def main(args):
    data = pd.read_csv(DATA_FILE, parse_dates=True, index_col=0, nrows=DATA_SIZE)

    raw_values = data.values[:, 0]

    # transform data to be stationary
    diff_values = difference(raw_values)
    # transform data to be supervised learning
    supervised_values = timeseries_to_supervised(diff_values).values
    # split data into train and test-sets
    train, test = supervised_values[0:-10], supervised_values[-10:]
    # transform the scale of the data
    train_scaled, test_scaled, scaler = scale(train, test)

    if os.path.exists(MODEL_FILE):
        # Load model
        lstm_model = load_model(MODEL_FILE)
    else:
        # Fit model
        lstm_model = fit_lstm(train_scaled, bath_size, epochs, units)

        # Save Model
        lstm_model.save(MODEL_FILE)

    # Forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    train_predict = lstm_model.predict(train_reshaped, batch_size=1)

    observe = raw_values[:-11]
    predict = train_predict.ravel()
    error = np.abs(observe - predict)

    print(type(observe))
    print(type(predict))
    print(observe.shape)
    print(predict.shape)
    print(type(error))
    print(error.shape)
    outliers = observe[percentile_based_outlier(error)]
    print(outliers)
    plt.plot(observe, color='b')
    plt.plot(predict, color='g')
    plt.plot(outliers, color='r')
    plt.show()
    sys.exit()

    # Test each time point
    test_scaled = scaler.transform(supervised_values[:, :])

    # Walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)

        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = invert_difference(raw_values, yhat, len(test_scaled) + 1 - i)

        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(test_scaled) + i + 1]

        print('Time=%s, Predicted=%f, Observed=%f' % (data.index[len(test_scaled) + i + 1], yhat, expected))

    # report performance
    rmse = np.sqrt(mean_squared_error(raw_values, predictions))
    print('RMSE: %.3f' % rmse)

    # line plot of observed vs predicted
    plt.plot(raw_values)
    plt.plot(predictions)
    plt.show()


def try_keras(args):
    data = pd.read_csv(DATA_FILE, parse_dates=True, index_col=0, nrows=500)

    X = data.ix[data.index[0:WINDOW_SIZE], 'v0']
    Y = data.ix[data.index[50:100], 'v0']

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    for i in range(round(len(data) / WINDOW_SIZE)):
        x1 = data.ix[data.index[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE], 'v0']
        x2 = data.ix[data.index[(i + 1) * WINDOW_SIZE:(i + 2) * WINDOW_SIZE], 'v0']

        # Fit the model
        model.fit(x1, x2, epochs=150, batch_size=10, verbose=1)

        # calculate predictions
        predictions = model.predict(x1)

        # evaluate the model
        # scores = model.evaluate(x1, x2)
        # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        predicted = [x[0] for x in predictions]
        df = DataFrame({"observed": X, "predict": predicted})
        rmse = np.sqrt(mean_squared_error(X, predicted))
        print("\nRMSE = %.3f" % rmse)
        # df.plot(kind='kde')


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv)

    elapsed = (time.time() - start)
    print("Time Usage: " + str(elapsed))
