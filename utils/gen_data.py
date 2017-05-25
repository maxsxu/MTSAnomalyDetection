#########################################################################
# --------->    FILE: gen.py
# --------->    AUTHOR: Max Xu
# --------->    MAIL: xuhuan@live.cn
# --------->    DATE: 05/03/2017    TIME:11:25:55
#########################################################################

# !/usr/bin/env python
# coding=utf-8

import os
import sys
import csv
import time
from datetime import datetime

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

DATE_START_TIME = "2010-1-1 10:00:00"
DATE_END_TIME = "2010-12-31 23:00:00"
DATE_FREQUENCY = "H"
TAG_NEGTIVE = "normal"
TAG_POSITIVE = "anormal"
OUTPUT_FILE = '../dataset/data.csv'

ANORMAL_COUNT = 3      # anormaly series count
ANORMAL_PERCENT = 20   # anormal_size = data_size / percent


def gen_0():
    with open(OUTPUT_FILE, 'w') as f:

        # timestamp
        timestamp = pd.date_range(DATE_START_TIME, DATE_END_TIME, freq=DATE_FREQUENCY)
        data_size = len(timestamp)

        # v0
        v0_data = np.random.uniform(0, 0.1, size=data_size) + 1

        # v1
        t = np.arange(0.0, float(data_size * 100), 0.01)
        v1_data = np.sin(2 * np.pi * t) + 1

        # headers = ["timestamp", "v0", "v1", "v2", "v3", "v4", "v5", "tag"]
        headers = ["timestamp", "v0", "v1", "tag"]

        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for (i, t) in enumerate(timestamp):
            d = [t]
            if i % 78 == 0:
                d.append(v0_data[i] + 1)
                d.append(v1_data[i] + 1)
                d.append(TAG_POSITIVE)
            else:
                d.append(v0_data[i])
                d.append(v1_data[i])
                d.append(TAG_NEGTIVE)
            d = dict(zip(headers, d))
            writer.writerow(d)

            print(d)


def gen_1():
    """
    np.sin generate wave. wave1 + wave2 + wave3
    :return: 
    """
    with open(OUTPUT_FILE, 'w') as f:
        # timestamp
        timestamp = pd.date_range(DATE_START_TIME, DATE_END_TIME, freq=DATE_FREQUENCY)
        data_size = len(timestamp)

        # tag
        tag = [TAG_NEGTIVE] * data_size

        # wave1 is normal (base)
        t = np.arange(0.0, float(data_size) / 100, 0.01)
        wave1 = np.sin(2 * 2 * np.pi * t) + 1
        noise = np.random.normal(0, 0.1, len(t))
        wave1 = wave1 + noise

        # wave2 is normal (base)
        wave2 = np.sin(2 * np.pi * t)

        # wave3 is anormal
        anormal_size = int(round(data_size / ANORMAL_PERCENT))
        t_anormal = np.arange(0.0, float(anormal_size) / 100, 0.01)
        wave3 = np.sin(10 * np.pi * t_anormal)

        # Randomly insert anomal
        for position in np.random.rand(ANORMAL_COUNT):
            insert = int(round(position * len(t)))
            wave1[insert:insert + anormal_size] = wave1[insert:insert + anormal_size] + wave3
            tag[insert:insert + anormal_size] = [TAG_POSITIVE] * anormal_size

        # Fixly insert anomal
        # insert = int(round(0.13 * len(t)))
        # wave1[insert:insert + anormal_size] = wave1[insert:insert + anormal_size] + wave3
        # tag[insert:insert + anormal_size] = [TAG_POSITIVE] * anormal_size
        #
        # insert = int(round(0.3 * len(t)))
        # wave1[insert:insert + anormal_size] = wave1[insert:insert + anormal_size] + wave3
        # tag[insert:insert + anormal_size] = [TAG_POSITIVE] * anormal_size
        #
        # insert = int(round(0.8 * len(t)))
        # wave1[insert:insert + anormal_size] = wave1[insert:insert + anormal_size] + wave3
        # tag[insert:insert + anormal_size] = [TAG_POSITIVE] * anormal_size

        # v0
        v0_data = wave1 + wave2

        # headers = ["timestamp", "v0", "v1", "v2", "v3", "v4", "v5", "tag"]
        headers = ["timestamp", "v0", "tag"]

        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for (i, t) in enumerate(timestamp):
            d = [t]
            d.append(v0_data[i])
            d.append(tag[i])
            d = dict(zip(headers, d))
            writer.writerow(d)

            print(d)


def main():
    gen_1()


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main()

    elapsed = (time.time() - start)
    print("Time Usage: " + str(elapsed))
