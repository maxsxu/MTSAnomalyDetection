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

from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.base import BaseEstimator, ClassifierMixin


class BiLSTMClassifier(BaseEstimator, ClassifierMixin):
    """Bidirectional LSTM Model for Binary Classification.

    """

    def __init__(self, input_shape, hidden_layers,
                 loss, optimizer,
                 epochs, batch_size,
                 verbose):
        """Initializing the classifier

        :param input_shape: tuple.
        :param hidden_layers: tuple. (units, ) the ith units is the total units of ith hidden layer.
                              All len(hidden_layers) hidden layers.
        :param loss: str.
        :param optimizer: str.
        :param epochs: int
        :param batch_size: int.
        :param verbose: int.
        """
        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.hidden_layers = hidden_layers

        # Construct model
        self.model = Sequential()

        for i, units in enumerate(self.hidden_layers):
            if i == 0:
                self.model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
            elif i == len(self.hidden_layers) - 1:
                self.model.add(Bidirectional(LSTM(units, return_sequences=False)))
            else:
                self.model.add(Bidirectional(LSTM(units, return_sequences=True)))

        self.model.add(Dense(1, activation='tanh'))

        # Configures the learning process.
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def fit(self, X, y):
        """Fit classifier.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        :param y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        :return: a trained LSTM model.
        """
        train_x = X.reshape(X.shape[0], 1, X.shape[1])
        train_y = y

        filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        self.model.fit(train_x, train_y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=0.05, callbacks=callbacks,
                       verbose=self.verbose)

        return self

    def predict(self, X):
        """Predict using the trained model

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        :return: pred_y.
        """
        test_x = X
        if len(X.shape) == 2:
            test_x = X.reshape(X.shape[0], 1, X.shape[1])

        pred_y = self.model.predict(test_x)

        pred_y = pred_y.round()
        pred_y = pred_y.ravel()
        pred_y = pred_y.astype('int64')

        return pred_y


if __name__ == "__main__":
    pass