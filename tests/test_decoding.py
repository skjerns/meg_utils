# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:05:05 2024

@author: Simon
"""
import sys; sys.path.append('../..')
from meg_utils import decoding

import unittest
import numpy as np

class TestDecoding(unittest.TestCase):

    def test_predict_proba_along_n_jobs(self):
        """n_jobs=1 and n_jobs=2 must return identical results."""
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        # X shape: (n_times=5, n_samples=20, n_features=10)
        X = rng.standard_normal((5, 20, 10))
        y = rng.integers(0, 3, size=20)
        clf = LogisticRegression(max_iter=200).fit(X[0], y)

        out1 = decoding.predict_proba_along(clf, X, axes=0, n_jobs=1)
        out2 = decoding.predict_proba_along(clf, X, axes=0, n_jobs=2)
        np.testing.assert_array_equal(out1, out2)
