#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-9-25

from __future__ import division  # for divide operation in python 2
from __future__ import print_function

import os
import sys
import random
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

matplotlib.use('Agg')

import numpy as np
import pandas as pd

from pandas import DataFrame

from scipy import stats

import sklearn.utils as utils
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

import matplotlib.font_manager
from matplotlib.patches import Ellipse, Circle
from matplotlib.offsetbox import (DrawingArea, AnnotationBbox)

try:
    from model import BiLSTMClassifier
    from processor import *
except Exception:
    from ensemble.model import BiLSTMClassifier
    from ensemble.processor import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

TIMESTEPS = 1
FEATURES = 9
TRAIN_PERCENTAGE = 0.7

EPOCHS = 60
BATCH_SIZE = 50


def unsupervised_methods():
    rng = np.random.RandomState(42)

    # Basic settings
    n_samples = 200
    outliers_fraction = 0.25
    clusters_separation = [0, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                     kernel="rbf", gamma=0.1),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(max_samples=n_samples,
                                            contamination=outliers_fraction,
                                            random_state=rng),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=outliers_fraction)}

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = -1

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)
        # Data generation
        X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
        X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
        X = np.r_[X1, X2]
        # Add outliers
        X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

        # Fit the model
        plt.figure(figsize=(9, 7))
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X)
                scores_pred = clf.negative_outlier_factor_
            else:
                clf.fit(X)
                scores_pred = clf.decision_function(X)
                y_pred = clf.predict(X)
            threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
            n_errors = (y_pred != ground_truth).sum()
            # plot the levels lines and the points
            if clf_name == "Local Outlier Factor":
                # decision_function is private for LOF
                Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            subplot = plt.subplot(2, 2, i + 1)
            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                             cmap=plt.cm.Blues_r)
            a = subplot.contour(xx, yy, Z, levels=[threshold],
                                linewidths=2, colors='red')
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')
            b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                                s=20, edgecolor='k')
            c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                                s=20, edgecolor='k')
            subplot.axis('tight')
            subplot.legend(
                [a.collections[0], b, c],
                ['learned decision function', 'true inliers', 'true outliers'],
                prop=matplotlib.font_manager.FontProperties(size=10),
                loc='lower right')
            subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
            subplot.set_xlim((-7, 7))
            subplot.set_ylim((-7, 7))
        plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
        plt.suptitle("Outlier detection")

    plt.show()


# For log prf
ds_precision = []
ds_recall = []
ds_f1 = []
header = []


def ensemble_classfier(data_file):
    file_name = data_file.split('/')[-1][:-4]

    dp = DataProcessor(data_file)

    # Create base/meta models
    metamodels = []

    metamodels.append(('BiLSTM', BiLSTMClassifier(input_shape=(1, dp.train_x.shape[1]),
                                                  hidden_layers=(64, 256, 100),
                                                  loss='mae', optimizer='rmsprop',
                                                  epochs=60, batch_size=72,
                                                  verbose=0)))

    metamodels.append(('Logistic Regression', LogisticRegression(class_weight='balanced',
                                                                 solver='liblinear',
                                                                 multi_class='ovr')))
    metamodels.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))

    metamodels.append(('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', )))
    metamodels.append(('SVM', SVC(kernel='poly')))
    metamodels.append(('Naive Bayes', GaussianNB()))
    metamodels.append(('CART', DecisionTreeClassifier()))

    ensemble = VotingClassifier(metamodels)
    ensemble = ensemble.fit(dp.train_x, dp.train_y)

    # Predict
    yhat = ensemble.predict(dp.test_x)

    # For P-R-F bar
    precision_list = []
    recall_list = []
    f1_list = []
    bar_labels = []

    # Evaluate
    precision = DataProcessor.precision_score(dp.test_y, yhat)
    recall = DataProcessor.recall_score(dp.test_y, yhat)
    f1 = DataProcessor.f1_score(dp.test_y, yhat)
    average_precision = metrics.average_precision_score(dp.test_y, yhat)
    precisions, recalls, _ = metrics.precision_recall_curve(dp.test_y, yhat)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    bar_labels.append("MTSAD")

    # Compared Models
    for (name, model) in metamodels:
        model = model.fit(dp.train_x, dp.train_y)
        pred_y = model.predict(dp.test_x)

        p = DataProcessor.precision_score(dp.test_y, pred_y)
        r = DataProcessor.recall_score(dp.test_y, pred_y)
        f = DataProcessor.f1_score(dp.test_y, pred_y)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        bar_labels.append(name)

    global ds_precision, ds_recall, ds_f1, header
    ds_precision.append([file_name] + precision_list)
    ds_recall.append([file_name] + recall_list)
    ds_f1.append([file_name] + f1_list)
    header = [''] + bar_labels

    # # Plot P-R-F1 Comparision bar
    # DataProcessor.plot_prf(file_name, precision_list, recall_list, f1_list, bar_labels)
    #
    # # Plot P-R curve
    # DataProcessor.plot_prc(file_name, precisions, recalls, average_precision)
    #
    # # Plot result
    # DataProcessor.plot_result(file_name, dp.train_x, dp.test_y, yhat,
    #                           precision, recall, f1)


def main(args):
    for root, dirs, files in os.walk("../dataset/10benchmark"):
        for f in files:
            data_file = os.path.join(root, f)
            print("\n# ", f)
            ensemble_classfier(data_file)

    DataProcessor.log_prf(ds_precision, ds_recall, ds_f1, header)


def draw_circle(ax, xy, radius):  # circle in the canvas coordinate
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
    from matplotlib.patches import Circle
    ada = AnchoredDrawingArea(20, 20, 0, 0,
                              loc=1, pad=0., frameon=False)
    p = Circle(xy, radius)
    ada.da.add_artist(p)
    ax.add_artist(ada)


def plot_outliers():
    data_file = "../dataset/10benchmark/pen-global.tab"
    dataset_name = data_file.split('/')[-1][:-4]
    df = pd.read_table(data_file, header=None)
    features = df.shape[1] - 1

    anormal_data = df[df.ix[:, features] == 'o']
    normal_data = df[df.ix[:, features] == 'n']
    fetched_normal_data = normal_data.ix[:normal_data.index[:len(anormal_data) * 5][-1], :]

    anormal_size = len(anormal_data)
    normal_size = len(normal_data)
    train_anormal_size = int(anormal_size * TRAIN_PERCENTAGE)
    train_normal_size = int(normal_size * TRAIN_PERCENTAGE)

    # Train
    train_data = pd.concat([anormal_data.ix[:train_anormal_size, :],
                            normal_data.ix[:train_normal_size, :]])
    train_data = utils.shuffle(train_data).reset_index(drop=True)

    # Test
    test_data = pd.concat([anormal_data.ix[train_anormal_size:, :],
                           normal_data.ix[train_normal_size:, :]])
    test_data = utils.shuffle(test_data).reset_index(drop=True)
    outliers_true = test_data[test_data.ix[:, features] == 'o']

    # test_data = test_data.ix[:150, :]
    # outliers_true = outliers_true.ix[:150, :]

    # Prediction
    encoder = LabelEncoder()
    train_data.ix[:, features] = encoder.fit_transform(train_data.ix[:, features])

    train_x = train_data.values[:, :features]
    train_y = train_data.values[:, features]

    test_x = test_data.values[:, :features]
    test_y = test_data.values[:, features]
    test_y = encoder.fit_transform(test_y)

    model = SVC(kernel='poly')
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    print("P=", DataProcessor.precision_score(test_y, pred_y))
    print("R=", DataProcessor.recall_score(test_y, pred_y))

    outliers = encoder.inverse_transform(pred_y.astype(int))
    pred_data = np.c_[test_x, outliers]
    pred_data = DataFrame(pred_data)

    outliers_pred = pred_data[pred_data.ix[:, features] == 'o']

    # print("Postition=", outliers_pred[pred_y != test_y].index)

    fig, axes = plt.subplots(features, 1, sharex=True)
    for i in range(features):
        ax = axes[i]
        ax.plot(test_data.ix[:, i], color='black')
        ax.set_ylabel("V{}".format(train_data.columns[i]))

        # Fix the display limits to see everything
        # ax.set_xlim(left=train_data.index.min(), right=train_data.index.max())
        # ax.set_ylim(bottom=train_data.ix[:, i].min(), top=train_data.ix[:, i].max())

        x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
        x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
        dx = x1 - x0
        dy = y1 - y0
        maxd = max(dx, dy)
        width = 2.5 * maxd / dx
        height = 2.5 * maxd / dy

        if i == 0:
            ax.set_title("MTSAD-V on {} dataset".format(dataset_name))

        if i == features - 1:
            ax.set_xlabel("Time")

        for x, y in zip(outliers_true.index, outliers_true.values[:, i]):
            xy = (x, y)

            # draw_circle(ax, xy, 1)

            circle = Ellipse(xy, width, height, fill=False, edgecolor='green')
            ax.add_patch(circle)

            # da = DrawingArea(0, 0, 0, 0)
            # circle = Circle(xy, 1, fill=False, edgecolor='green')
            # da.add_artist(circle)
            # ab = AnnotationBbox(da, xy,
            #                     xycoords='data',
            #                     boxcoords=("axes fraction", "data"),
            #                     box_alignment=(0., 0.),
            #                     pad=0.1,
            #                     arrowprops=dict(arrowstyle="-"))
            # ax.add_artist(ab)

            # ax.annotate("O", xy=xy,
            #             bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.5),
            #             arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))

        for x, y in zip(outliers_pred.index, outliers_pred.values[:, i]):
            xy = (x + 0.2, y + 0.2)

            # draw_circle(ax, xy, 1.2)

            circle = Ellipse(xy, width, height, fill=False, edgecolor='red')
            ax.add_patch(circle)

            # da = DrawingArea(0, 0, 0, 0)
            # circle = Circle(xy, 1.2, fill=False, edgecolor='red')
            # da.add_artist(circle)
            # ab = AnnotationBbox(da, xy,
            #                     xycoords='data',
            #                     boxcoords=("axes fraction", "data"),
            #                     box_alignment=(0., 0.),
            #                     pad=0.1,
            #                     arrowprops=dict(arrowstyle="-"))
            # ax.add_artist(ab)

            # ax.annotate("O", xy=xy,
            #             bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.5),
            #             arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))

    plt.grid()
    plt.show()


if __name__ == "__main__":
    # main(sys.argv)

    plot_outliers()

    # from matplotlib.patches import Circle
    # from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, OffsetBox,
    #                                   AnnotationBbox)
    # from matplotlib.cbook import get_sample_data
    #
    # fig, ax = plt.subplots()
    #
    # # Define a 2nd position to annotate (don't display with a marker this time)
    # xy = [0.3, 0.55]
    # ax.plot(xy[0], xy[1], ".r")
    #
    # # Annotate the 2nd position with a circle patch
    # da = DrawingArea(0, 0, 0, 0)
    # p = Circle(xy, 10, fill=False, edgecolor='red')
    # da.add_artist(p)
    #
    # ab = AnnotationBbox(da, xy,
    #                     xycoords='data',
    #                     boxcoords=("axes fraction", "data"),
    #                     box_alignment=(0., 0.),
    #                     pad=0.1,
    #                     arrowprops=dict(arrowstyle="-"))
    #
    # ax.add_artist(ab)
    #
    #
    # # Fix the display limits to see everything
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    #
    # plt.show()
