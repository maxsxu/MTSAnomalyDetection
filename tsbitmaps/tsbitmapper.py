#!/usr/bin/env python
# coding=utf-8

import numpy as np
from collections import defaultdict


class TSBitMapper:
    """
    
    Implements Time-series Bitmap model for anomaly detection
    
    Based on the papers "Time-series Bitmaps: A Practical Visualization Tool for working with Large Time Series Databases"
    and "Assumption-Free Anomaly Detection in Time Series"
    
    Test data and parameter settings taken from http://alumni.cs.ucr.edu/~wli/SSDBM05/
    """

    def __init__(self, feature_window_size=None, bins=5, level_size=3, lag_window_size=None, lead_window_size=None,
                 q=99.7):

        """
        
        :param int feature_window_size: should be about the size at which events happen
        :param int or array-like bins: a scalar number of equal-width bins or a 1-D and monotonic array of bins.
        :param int level_size: desired level of recursion of the bitmap
        :param int lag_window_size: how far to look back, None for supervised learning
        :param int lead_window_size: how far to look ahead
        :param float in range of [0,100] q: the qth percentile as the threshold for anomalies
        """

        assert feature_window_size > 0, 'feature_window_size must be a positive integer'
        assert lead_window_size >= feature_window_size, 'lead_window_size must be >= feature_window_size'

        # bitmap parameters
        self._feature_window_size = feature_window_size

        self._level_size = level_size

        self._lag_window_size = lag_window_size

        self._lead_window_size = lead_window_size

        self._bins = bins
        self._num_bins = self._get_num_bins(bins)
        self._q = q
        self._bitmap_scores = None

    def _get_num_bins(self, bins):
        if np.isscalar(bins):
            num_bins = bins
        else:
            num_bins = len(bins)  # bins is an array of bins
        return num_bins

    def discretize(self, ts, bins=None, global_min=None, global_max=None):
        if bins is None:
            bins = self._bins

        if np.isscalar(bins):
            num_bins = bins

            min_value = ts.min()
            max_value = ts.max()
            if min_value == max_value:
                min_value = global_min
                max_value = global_max
            step = (max_value - min_value) / num_bins
            ts_bins = np.arange(min_value, max_value, step)
        else:
            ts_bins = bins

        inds = np.digitize(ts, ts_bins)
        binned_ts = tuple(str(i - 1) for i in inds)
        return binned_ts

    def discretize_by_feat_window(self, ts, bins=None, feature_window_size=None):
        if bins is None:
            bins = self._bins

        if feature_window_size is None:
            feature_window_size = self._feature_window_size

        n = len(ts)
        windows = ()
        global_min = ts.min()
        global_max = ts.max()

        for i in range(0, n - n % feature_window_size, feature_window_size):
            binned_fw = self.discretize(ts[i: i + feature_window_size], bins, global_min, global_max)
            windows += binned_fw
        if n % feature_window_size > 0:
            last_binned_fw = self.discretize(ts[- (n % feature_window_size):], bins, global_min, global_max)
            windows += last_binned_fw

        return windows

    def get_tsbitmap(self, ts, with_feat_window=True, level_size=None, step=None):
        binned_ts = self.discretize(ts)
        tsbitmap = {}
        if with_feat_window:
            tsbitmap = self.get_bitmap_with_feat_window(binned_ts)
        else:
            tsbitmap = self.get_bitmap(binned_ts)
        return tsbitmap

    def get_bitmap(self, chunk, level_size=None):
        """
        
        :param str chunk: symbol sequence representation of a univariate time series
        :param int level_size: desired level of recursion of the bitmap
        :return: bitmap representation of `chunk`
        """
        bitmap = defaultdict(int)
        n = len(chunk)
        if level_size is None:
            level_size = self._level_size
        for i in range(n):
            if i <= n - level_size:
                feat = chunk[i: i + level_size]  # I think feat is for "feature". -max
                bitmap[feat] += 1  # frequency count
        max_freq = max(bitmap.values())
        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap

    def get_bitmap_with_feat_window(self, chunk, level_size=None, step=None):
        """
        
        :param str chunk: symbol sequence representation of a univariate time series
        :param int level_size: desired level of recursion of the bitmap
        :param int step: length of the feature window
        :return: bitmap representation of `chunk`
        """
        if step is None:
            step = self._feature_window_size
        if level_size is None:
            level_size = self._level_size

        bitmap = defaultdict(int)
        n = len(chunk)

        for i in range(0, n - n % step, step):
            for j in range(step - level_size + 1):
                feat = chunk[i + j: i + j + level_size]
                bitmap[feat] += 1  # frequency count

        if n % step > 0:
            for i in range(n - n % step, n - level_size + 1):
                feat = chunk[i: i + level_size]
                bitmap[feat] += 1

        max_freq = max(bitmap.values())

        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap

    def _slide_lead_chunks(self, ts):
        """
        In supervised training, the entire training time series as the reference data can be used as the lag window.

        """

        binned_ts = self.discretize_by_feat_window(ts)
        ts_len = len(binned_ts)

        scores = np.zeros(len(ts))

        leadws = self._lead_window_size
        featws = self._level_size

        lead_bitmap = self.get_bitmap_with_feat_window(binned_ts[0: leadws])
        egress_lead_feat = binned_ts[0: featws]

        for i in range(1, ts_len - leadws + 1):
            lead_chunk = binned_ts[i: i + leadws]
            ingress_lead_feat = lead_chunk[-featws:]

            lead_bitmap[ingress_lead_feat] += 1
            lead_bitmap[egress_lead_feat] -= 1

            scores[i] = self.bitmap_distance(self._ref_ts_bitmap, lead_bitmap)

            egress_lead_feat = lead_chunk[0: featws]

        return scores

    def _slide_chunks(self, ts):
        """

        Args:
            ts (1-D numpy array or pandas.Series): Unidimensional Time Series (UTS).

        Returns:
            1-D ndarray: scores of UTS.

        """
        lag_bitmap = {}
        lead_bitmap = {}
        scores = np.zeros(len(ts))

        egress_lag_feat = ()
        egress_lead_feat = ()

        # to SAX
        binned_ts = self.discretize_by_feat_window(ts)
        ts_len = len(binned_ts)

        lagws = self._lag_window_size
        leadws = self._lead_window_size
        featws = self._level_size

        for i in range(lagws, ts_len - leadws + 1):

            if i == lagws:
                lag_chunk = binned_ts[i - lagws: i]
                lead_chunk = binned_ts[i: i + leadws]

                # get Bitmap (caculate the frequency, standardization)
                lag_bitmap = self.get_bitmap_with_feat_window(lag_chunk)
                lead_bitmap = self.get_bitmap_with_feat_window(lead_chunk)

                # caculate anomaly score
                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

                egress_lag_feat = lag_chunk[0: featws]
                egress_lead_feat = lead_chunk[0: featws]

            else:

                lag_chunk = binned_ts[i - lagws: i]
                lead_chunk = binned_ts[i: i + leadws]

                ingress_lag_feat = lag_chunk[-featws:]
                ingress_lead_feat = lead_chunk[-featws:]

                lag_bitmap[ingress_lag_feat] += 1
                lag_bitmap[egress_lag_feat] -= 1

                lead_bitmap[ingress_lead_feat] += 1
                lead_bitmap[egress_lead_feat] -= 1

                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

                egress_lag_feat = lag_chunk[0: featws]
                egress_lead_feat = lead_chunk[0: featws]

        return scores

    def bitmap_distance(self, lag_bitmap, lead_bitmap):
        """
        Computes the dissimilarity of two bitmaps.
        """
        dist = 0
        lag_feats = set(lag_bitmap.keys())
        lead_feats = set(lead_bitmap.keys())
        shared_feats = lag_feats & lead_feats

        for feat in shared_feats:
            dist += (lead_bitmap[feat] - lag_bitmap[feat]) ** 2

        for feat in lag_feats - shared_feats:
            dist += lag_bitmap[feat] ** 2

        for feat in lead_feats - shared_feats:
            dist += lead_bitmap[feat] ** 2

        return dist

    def fit(self, ts):
        """Computes the reference bitmaps of a univariate time series `ts`.

        Args:
            ts (1-D numpy array or pandas.Series):

        """
        if self._lag_window_size is None:
            assert len(ts) >= self._lead_window_size, 'sequence length must be >= than lead_window_size'
        else:
            assert len(
                ts) >= self._lag_window_size + self._lead_window_size, 'sequence length must be >= (lag_window_size + lead_window_size)'

        self._ref_ts = ts
        self._ref_ts_bitmap = self.get_bitmap_with_feat_window(self.discretize(ts))

    def fit_predict(self, ts):
        """ Unsupervised training of TSBitMaps.

        Args:
            ts (1-D numpy array or pandas.Series): Unidimensional Time Series.

        Returns:
            labels: `+1` for anomaly observations and `-1` for normal observations.

        """
        assert self._lag_window_size > self._feature_window_size, 'lag_window_size must be >= feature_window_size'

        self._ref_ts = ts
        scores = self._slide_chunks(ts)
        self._ref_bitmap_scores = scores

        thres = np.percentile(scores[self._lag_window_size: -self._lead_window_size + 1], self._q)

        labels = np.full(len(ts), -1)
        for idx, score in enumerate(scores):
            if score > thres:
                labels[idx] = 1

        return labels

    def predict(self, ts):
        """Predict if a time series contains outliers or not.

        Args:
            ts (1-D numpy array or pandas.Series):

        Returns:
            labels: `+1` for anomaly observations and `-1` for normal observations.

        """
        cur_scores = None
        labels = np.full(len(ts), -1)

        if (len(self._ref_ts) != len(ts)):
            cur_scores = self._slide_lead_chunks(ts)
        elif (np.allclose(self._ref_ts, ts)):
            labels = self.fit_predict(ts)
        else:
            cur_scores = self._slide_lead_chunks(ts)

        if cur_scores is not None:
            self._bitmap_scores = cur_scores

            thres = np.percentile(cur_scores[0: -self._lead_window_size + 1], self._q)

            for idx, score in enumerate(cur_scores):
                if score > thres:
                    labels[idx] = 1

        return labels

    def get_bitmap_scores(self):
        if self._bitmap_scores is None:
            return self._ref_bitmap_scores
        else:
            return self._bitmap_scores

    def get_ref_bitmap_scores(self):
        return self._ref_bitmap_scores
