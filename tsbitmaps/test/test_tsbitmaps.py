# coding=utf-8

import os
import sys
import numpy as np
import pandas as pd
import unittest

sys.path.append(os.path.dirname(__file__))

from tsbitmaps.tsbitmapper import TSBitMapper
from tsbitmaps.bitmapviz import create_bitmap_grid


class TestBitmapAlgorithm(unittest.TestCase):
    def test_bitmap(self):
        bmp = TSBitMapper(feature_window_size=5, bins=8, level_size=2,
                          lag_window_size=10, lead_window_size=10, q=95)
        x = np.random.rand(500)
        binned_x = bmp.discretize(x)

        self.assertEqual(len(binned_x), len(x))
        self.assertTrue(set(binned_x) == set('01234567'))

        symbol_seq = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2', '3')  # '01234567890123'
        sample_bitmap = bmp.get_bitmap(symbol_seq)
        self.assertEqual(len(sample_bitmap), 10)
        self.assertTrue(('4', '5') in sample_bitmap.keys())
        self.assertTrue(('9', '0') in sample_bitmap.keys())
        self.assertEqual(sample_bitmap['0', '1'], 1)

        sample_bitmap_w = bmp.get_bitmap_with_feat_window(symbol_seq)
        self.assertEqual(len(sample_bitmap_w), 8)
        self.assertTrue(('4', '5') not in sample_bitmap_w.keys())
        self.assertTrue(('9', '0') not in sample_bitmap_w.keys())
        self.assertEqual(sample_bitmap_w[('0', '1')], 1)

        ypred = bmp.fit_predict(x)
        scores = bmp.get_ref_bitmap_scores()
        self.assertTrue((scores[0:bmp._lag_window_size] == 0.0).all())
        self.assertTrue((scores[bmp._lag_window_size:-bmp._lead_window_size] >= 0).all())
        self.assertTrue(0 < (ypred == -1).sum() <= 25)

    def test_anomaly_detection_ecg(self):
        ecg_norm = np.loadtxt('data/ecg_normal.txt')
        ecg_anom = np.loadtxt('data/ecg_anom.txt')

        bmp = TSBitMapper(feature_window_size=20, bins=5, level_size=3, lag_window_size=200, lead_window_size=40)
        ypred_unsupervised = bmp.fit_predict(ecg_anom)
        self.assertTrue(0 < (ypred_unsupervised == -1).sum() <= 3)

        bmp.fit(ecg_norm)
        ypred_supervised = bmp.predict(ecg_anom)
        self.assertTrue(0 < (ypred_supervised == -1).sum() <= 3)

    def test_anomaly_detection_pattern(self):
        pattern_norm = np.loadtxt('data/pattern_normal.txt')
        pattern_anom = pd.read_csv('data/pattern_anom.txt').iloc[:, 0]

        bmp = TSBitMapper(feature_window_size=50, bins=5, level_size=2, lag_window_size=200, lead_window_size=100)
        ypred_unsupervised = bmp.fit_predict(pattern_anom)
        self.assertTrue(0 < (ypred_unsupervised == -1).sum() <= 3)

        bmp.fit(pattern_norm)
        ypred_supervised = bmp.predict(pattern_anom)
        self.assertTrue(0 < (ypred_supervised == -1).sum() <= 3)

    # @unittest.skip("tmp")
    def test_bitmapviz(self):
        bmp = TSBitMapper(feature_window_size=20, bins=12, level_size=3, lag_window_size=200, lead_window_size=40)
        ecg_anom = np.loadtxt('data/ecg_anom.txt')
        ecg_bitmap = bmp.get_tsbitmap(ecg_anom)
        bmp_grid = create_bitmap_grid(ecg_bitmap, n=4, num_bins=12, level_size=3)
        self.assertEqual((bmp_grid > 0).sum(), len(ecg_bitmap))
        self.assertEqual(bmp_grid.shape, (27, 64))


if __name__ == '__main__':
    unittest.main()
