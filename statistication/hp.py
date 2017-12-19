#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-10-31

"""
Anomaly Detection (ad) Using hp filter and mad test
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib.pyplot as plt


# Hodrick Prescott filter
def hp_filter(x, lamb=5000):
    w = len(x)
    b = [[1] * w, [-2] * w, [1] * w]
    D = sparse.spdiags(b, [0, 1, 2], w - 2, w)
    I = sparse.eye(w)
    B = (I + lamb * (D.transpose() * D))
    return sparse.linalg.dsolve.spsolve(B, x)


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def AnomalyDetection(x, alpha=0.2, lamb=5000):
    """
    x         : pd.Series
    alpha     : The level of statistical significance with which to
                accept or reject anomalies. (expon distribution)
    lamb      : penalize parameter for hp filter
    return r  : Data frame containing the index of anomaly
    """
    # calculate residual
    xhat = hp_filter(x, lamb=lamb)
    resid = x - xhat

    # drop NA values
    ds = pd.Series(resid)
    ds = ds.dropna()

    # Remove the seasonal and trend component,
    # and the median of the data to create the univariate remainder
    md = np.median(x)
    data = ds - md

    # process data, using median filter
    ares = (data - data.median()).abs()
    data_sigma = data.mad() + 1e-12
    ares = ares / data_sigma

    # compute significance
    p = 1. - alpha
    R = stats.expon.interval(p, loc=ares.mean(), scale=ares.std())
    threshold = R[1]

    # extract index, np.argwhere(ares > md).ravel()
    r_id = ares.index[ares > threshold]

    return r_id


# demo
def main(args):
    # fix
    np.random.seed(42)

    # sample signals
    N = 1024  # number of sample points
    t = np.linspace(0, 2 * np.pi, N)
    y = np.sin(t) + 0.02 * np.random.randn(N)

    # outliers are assumed to be step/jump events at sampling points
    M = 3  # number of outliers
    for ii, vv in zip(np.random.rand(M) * N, np.random.randn(M)):
        y[int(ii):] += vv

    # detect anomaly
    r_idx = AnomalyDetection(y, alpha=0.1)

    # plot the result
    plt.figure()
    plt.plot(y, 'b-')
    plt.plot(r_idx, y[r_idx], 'ro')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
