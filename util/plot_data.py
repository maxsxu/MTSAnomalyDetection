#########################################################################
# --------->    FILE: plot_data.py
# --------->    AUTHOR: Max Xu
# --------->    MAIL: xuhuan@live.cn
# --------->    DATE: 05/04/2017    TIME:00:09:41
#########################################################################

#!/usr/bin/env python
# coding=utf-8

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Series, DataFrame


def main(args):
    if len(args) < 1:
        print("Usage: plot_data.py data.csv [rows] \n")
        sys.exit()

    data_file = args[0]
    size = None
    if len(args) > 1:
        size = int(args[1])

    data = pd.read_csv(data_file, index_col='timestamp', nrows=size)

    data.plot()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
