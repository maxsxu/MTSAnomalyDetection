# coding=utf-8

# Created by max on 17-12-11

import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA


def main(args):
    df = pd.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',')

    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True)  # drops the empty line at file-end

    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    # Standardizing
    X_std = StandardScaler().fit_transform(X)

    # Covariance Matrix
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)

    # Eigendecomposition
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # SVD
    u, s, v = np.linalg.svd(X_std.T)

    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                          eig_pairs[1][1].reshape(4, 1)))

    print('Matrix W:\n', matrix_w)

    # 3. Projection Onto the New Feature Space
    Y = X_std.dot(matrix_w)


    sklearn_pca = sklearnPCA(n_components=1)
    Y_sklearn = sklearn_pca.fit_transform(X_std)


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("Used {0:0.3f} seconds".format(elapsed))
