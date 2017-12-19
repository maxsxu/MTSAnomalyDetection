#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-9-15

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def f1_score(predictions, y):
    """F_1Score

    Args:
        predictions 预测
        y 真实值
    Returns:
        F_1Score
    """
    TP = np.sum((predictions == 1) & (y == 1))
    FP = np.sum((predictions == 1) & (y == 0))
    FN = np.sum((predictions == 0) & (y == 1))
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)
    if precision + recall == 0:
        return 0
    else:
        return (2.0 * precision * recall) / (precision + recall)


def gaussianModel(X):
    """高斯模型

    Args:
        X 样本集
    Returns:
        p 模型
    """
    # 参数估计
    m, n = X.shape
    mu = np.mean(X, axis=0)
    delta2 = np.var(X, axis=0)

    def p(x):
        """p(x)

        Args:
            x x
            mu mu
            delta2 delta2
        Returns:
            p
        """
        total = 1
        for j in range(x.shape[0]):
            total *= np.exp(-np.power((x[j, 0] - mu[0, j]), 2) / (2 * delta2[0, j] ** 2)
                            ) / (np.sqrt(2 * np.pi * delta2[0, j]))
        return total

    return p


def multivariateGaussianModel(X):
    """多元高斯模型

    Args:
        X 样本集
    Returns:
        p 模型
    """
    # 参数估计
    m, n = X.shape
    mu = np.mean(X.T, axis=1)
    Sigma = np.var(X, axis=0)
    Sigma = np.diagflat(Sigma)
    # Sigma = np.mat(np.cov(X.T))
    detSigma = np.linalg.det(Sigma)

    def p(x):
        """p(x)

        Args:
            x x
            mu mu
            delta2 delta2
        Returns:
            p
        """
        x = x - mu
        return np.exp(-x.T * np.linalg.pinv(Sigma) * x / 2).A[0] * \
               ((2 * np.pi) ** (-n / 2) * (detSigma ** (-0.5)))

    return p


def train(X, model=gaussianModel):
    """训练函数

    Args:
        X 样本集
    Returns:
        p 概率模型
    """
    return model(X)


def selectEpsilon(XVal, yVal, p):
    # 通过交叉验证集，选择最好的 epsilon 参数
    pVal = np.mat([p(x.T) for x in XVal]).reshape(-1, 1)
    step = (np.max(pVal) - np.min(pVal)) / 1000
    bestEpsilon = 0
    bestF1 = 0
    for epsilon in np.arange(np.min(pVal), np.max(pVal), step):
        predictions = pVal < epsilon
        f1 = f1_score(predictions, yVal)
        if f1 > bestF1:
            bestF1 = f1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


# 小维度测试......
data = loadmat('ex8data1.mat')
X = np.mat(data['X'])
XVal = np.mat(data['Xval'])
yVal = np.mat(data['yval'])

# p = anomaly.train(X)
p = train(X, model=multivariateGaussianModel)
pTest = np.mat([p(x.T) for x in X]).reshape(-1, 1)

# 绘制数据点
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.plot(X[:, 0], X[:, 1], 'bx')
epsilon, f1 = selectEpsilon(XVal, yVal, p)

print('Best epsilon found using cross-validation: %e\n' % epsilon)
print('Best F1 on Cross Validation Set:  %f\n' % f1)
print('# Outliers found: %d' % np.sum(pTest < epsilon))

# 获得训练集的异常点
outliers = np.where(pTest < epsilon, True, False).ravel()
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', lw=2, markersize=10, fillstyle='none', markeredgewidth=1)
n = np.linspace(0, 35, 100)
X1 = np.meshgrid(n, n)
XFit = np.mat(np.column_stack((X1[0].T.flatten(), X1[1].T.flatten())))
pFit = np.mat([p(x.T) for x in XFit]).reshape(-1, 1)
pFit = pFit.reshape(X1[0].shape)
if not np.isinf(np.sum(pFit)):
    plt.contour(X1[0], X1[1], pFit, 10.0 ** np.arange(-20, 0, 3).T)
plt.show()

# 大维度测试......
data = loadmat('ex8data2.mat')
X = np.mat(data['X'])
XVal = np.mat(data['Xval'])
yVal = np.mat(data['yval'])

# p = anomaly.train(X)
p = train(X, model=multivariateGaussianModel)
pTest = np.mat([p(x.T) for x in X]).reshape(-1, 1)

epsilon, f1 = selectEpsilon(XVal, yVal, p)

print('Best epsilon found using cross-validation: %e\n' % epsilon)
print('Best F1 on Cross Validation Set:  %f\n' % f1)
print('# Outliers found: %d' % np.sum(pTest < epsilon))
