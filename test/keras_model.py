#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-4

import sys
import numpy

import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("dataset/pima-indians-diabetes.data.txt", delimiter=",")
predict_rounded = []


def main(args):
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=2)

    # calculate predictions
    predictions = model.predict(X)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # round predictions
    predict_rounded = [round(x[0]) for x in predictions]
    print(predict_rounded)


def evaluate_predict():
    # Prediction Evaluation
    observed = dataset[:, 8]
    df = DataFrame({"observed": observed, "predict": predict_rounded})

    mse = np.sum(np.power(observed - predict_rounded, 2)) / len(observed)
    print("\nMSE = %.3f" % mse)

    df.plot(kind='kde')


if __name__ == "__main__":
    main(sys.argv)