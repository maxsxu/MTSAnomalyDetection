# coding=utf-8

import numpy as np
import math
from pprint import pprint

def create_bitmap_grid(bitmap, n, num_bins, level_size):
    """
    Arranges a time-series bitmap into a 2-D grid for heatmap visualization
    """
    assert num_bins % n == 0, 'num_bins has to be a multiple of n'
    m = num_bins // n

    row_count = int(math.pow(m, level_size))
    col_count = int(math.pow(n, level_size))

    grid = np.full((row_count, col_count), 0.0)

    for feat, count in bitmap.items():
        i, j = symbols2index(m, n, feat)
        grid[i, j] = count
    return grid


def create_unit_grid(m, n):
    """
    
    :param m: row count
    :param n: column count
    :return unit_grid: array of shape `(m,n)` contains all integers in `[0, m*n - 1]`
    """
    unit_grid = np.ndarray(shape=(m, n), dtype=int)
    pprint(unit_grid)
    for i in range(0, m):
        for j in range(0, n):
            unit_grid[i, j] = i * n + j

    return unit_grid


def num2index(n, s):
    i = s // n
    j = s % n
    return i, j


def symbols2index(m, n, feat):
    level_size = len(feat)
    i = 0
    j = 0
    for k in range(0, level_size):
        cur_i, cur_j = num2index(n, int(feat[k]))
        i += int(cur_i * math.pow(m, level_size - k - 1))
        j += int(cur_j * math.pow(n, level_size - k - 1))
    return i, j


def index2symbos(m, n, i, j, level_size):
    feat_arr = np.ndarray(level_size, dtype=int)
    for k in range(0, level_size):
        cur_i = int(i // math.pow(m, level_size - k - 1))
        cur_j = int(j // math.pow(n, level_size - k - 1))

        feat_arr[k] = cur_i * m + cur_j
        i = i % math.pow(m, level_size - k - 1)
        j = j % math.pow(n, level_size - k - 1)
    feat = tuple(str(symbol) for symbol in feat_arr)
    return feat


if __name__ == '__main__':
    pprint(create_unit_grid(3, 5))
