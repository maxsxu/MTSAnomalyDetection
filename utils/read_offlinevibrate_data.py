#########################################################################
# --------->    FILE: read_offlinevibrate_data.py
# --------->    AUTHOR: Max Xu
# --------->    MAIL: xuhuan@live.cn
# --------->    DATE: 05/04/2017    TIME:00:09:41
#########################################################################

#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame


def main(args):
    data = pd.read_csv("OfflineVibrateData.csv", engine='c', nrows=100000, usecols=[0, 11, 16, 18, 19, 20], index_col=['设备ID', '采集时间'])
    data.to_csv("t.csv")
    

if __name__ == "__main__":
    main(sys.argv)
