#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-10-10


from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import random
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

matplotlib.use('Agg')

import matplotlib.font_manager

import numpy as np
import pandas as pd

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

try:
    from model import BiLSTMClassifier
    from processor import *
except Exception:
    from ensemble.model import BiLSTMClassifier
    from ensemble.processor import *


def stacking(data_file):
    # Stacking base models
    base_models = []
    base_models.append(('lr', LogisticRegression()))
    base_models.append(('lda', LinearDiscriminantAnalysis()))
    base_models.append(('knn', KNeighborsClassifier()))
    base_models.append(('svm', SVC(kernel='poly')))
    base_models.append(('gnb', GaussianNB()))
    base_models.append(('cart', DecisionTreeClassifier()))

    # Create stacking result file
    stacking_data_file = data_file
    if not data_file[5:].startswith('stacking_'):
        stacking_data_file = data_file[:5] + "stacking_" + data_file[5:]

        dp = DataProcessor(data_file)

        # Stacking result
        new_train_data = np.c_[dp.test_y]

        for (_, model) in base_models:
            model.fit(dp.train_x, dp.train_y)
            pred_y = model.predict(dp.test_x)

            new_train_data = np.c_[pred_y, new_train_data]

        np.savetxt(stacking_data_file, new_train_data, fmt='%d', delimiter='\t')

    # new train and test
    dp = DataProcessor(stacking_data_file)

    fig_file = stacking_data_file[5:-4]
    precision_list = []
    recall_list = []
    f1_list = []
    bar_labels = []

    for (name, model) in base_models:
        model.fit(dp.train_x, dp.train_y)
        pred_y = model.predict(dp.test_x)

        p = metrics.precision_score(dp.test_y, pred_y)
        print("p=", p)
        r = metrics.recall_score(dp.test_y, pred_y)
        print("r=", r)
        f = metrics.f1_score(dp.test_y, pred_y)
        print("f1=", f)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        bar_labels.append(name)

    # Plot P-R-F1 Comparision bar
    DataProcessor.plot_prf(fig_file, precision_list, recall_list, f1_list, bar_labels)


def main(args):
    for root, dirs, files in os.walk("data"):
        for f in files:
            data_file = os.path.join(root, f)
            print("\n# ", f)
            stacking(data_file)


if __name__ == "__main__":
    main(sys.argv)
