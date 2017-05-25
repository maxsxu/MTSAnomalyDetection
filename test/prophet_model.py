#!/usr/bin/env python
# coding=utf-8

# Created by max on 17-5-4.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fbprophet import Prophet
from pandas import Series, DataFrame


DATA_FILE = "dataset/data0.csv"

def main(args):
    data = pd.read_csv(DATA_FILE, parse_dates=True, index_col='timestamp')

    # Re-group data to fit for Prophet data format
    data['ds'] = data.index
    data = data.reindex(columns=['ds', 'v0', 'v1', 'result'])
    data = data.rename(columns={"v0": 'y'})

    model = Prophet()
    model.fit(data.ix[data.index[0:500]])

    future = model.make_future_dataframe(120, 'H')
    forecast = model.predict(future)

    model.plot(forecast)
    model.plot_components(forecast)

    plt.show()


if __name__ == "__main__":
    main(sys.argv)