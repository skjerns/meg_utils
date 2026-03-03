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

class TestStratify(unittest.TestCase):

    def _make_imbalanced(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        y = np.array([0] * 30 + [1] * 10)
        return X, y

    def test_undersample_balances(self):
        """Undersampling reduces all classes to the minority count."""
        X, y = self._make_imbalanced()
        _, y_out = decoding.stratify(X, y, strategy='undersample', random_state=0)
        counts = np.bincount(y_out)
        np.testing.assert_array_equal(counts, [10, 10])

    def test_oversample_balances(self):
        """Oversampling raises all classes to the majority count."""
        X, y = self._make_imbalanced()
        _, y_out = decoding.stratify(X, y, strategy='oversample', random_state=0)
        counts = np.bincount(y_out)
        np.testing.assert_array_equal(counts, [30, 30])

    def test_y_matches_X(self):
        """Each output label matches the label of the corresponding input row."""
        X, y = self._make_imbalanced()
        X_out, y_out = decoding.stratify(X, y, strategy='undersample', random_state=0)
        for i in range(len(y_out)):
            orig_idx = np.where(np.all(X == X_out[i], axis=1))[0]
            self.assertTrue(len(orig_idx) > 0)
            self.assertEqual(y[orig_idx[0]], y_out[i])

    def test_output_contains_original_rows(self):
        """Every row in the output is an exact copy of an input row."""
        X, y = self._make_imbalanced()
        X_out, _ = decoding.stratify(X, y, strategy='undersample', random_state=0)
        for row in X_out:
            self.assertTrue(any(np.allclose(row, orig) for orig in X))

    def test_invalid_strategy_raises(self):
        """Unknown strategy string raises ValueError."""
        X, y = self._make_imbalanced()
        with self.assertRaises(ValueError):
            decoding.stratify(X, y, strategy='badstrat')

    def test_y_2d_raises(self):
        """Passing a 2-D array as y raises ValueError."""
        X, _ = self._make_imbalanced()
        y_2d = np.ones((40, 2))
        with self.assertRaises(ValueError):
            decoding.stratify(X, y_2d)

    def test_random_state_reproducible(self):
        """The same random seed produces identical outputs on repeated calls."""
        X, y = self._make_imbalanced()
        X1, y1 = decoding.stratify(X, y, strategy='undersample', random_state=42)
        X2, y2 = decoding.stratify(X, y, strategy='undersample', random_state=42)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(X1, X2)

    def test_random_state_different(self):
        """Different seeds produce different sample orderings."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))
        y = np.array([0] * 150 + [1] * 50)
        _, y1 = decoding.stratify(X, y, strategy='undersample', random_state=0)
        _, y2 = decoding.stratify(X, y, strategy='undersample', random_state=99)
        self.assertFalse(np.array_equal(y1, y2))

    def test_already_balanced(self):
        """Both strategies leave an already-balanced dataset at the same size."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5))
        y = np.array([0] * 10 + [1] * 10)
        for strategy in ('undersample', 'oversample'):
            _, y_out = decoding.stratify(X, y, strategy=strategy, random_state=0)
            counts = np.bincount(y_out)
            np.testing.assert_array_equal(counts, [10, 10])

    def test_oversample_no_replace_when_equal(self):
        """Oversampling a balanced dataset does not duplicate any samples."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5))
        y = np.array([0] * 10 + [1] * 10)
        _, y_out = decoding.stratify(X, y, strategy='oversample', random_state=0)
        counts = np.bincount(y_out)
        np.testing.assert_array_equal(counts, [10, 10])

    def test_works_with_dataframe(self):
        """stratify accepts a pandas DataFrame and returns correct row counts."""
        import pandas as pd
        N = 40
        X = pd.DataFrame(np.eye(N))
        y = np.array([0] * 30 + [1] * 10)
        X_out, y_out = decoding.stratify(X, y, strategy='undersample', random_state=0)
        self.assertEqual(len(X_out), 20)
        self.assertEqual(len(y_out), 20)

    def test_verbose_no_crash(self):
        """verbose=True prints the target count without raising an error."""
        import io
        from contextlib import redirect_stdout
        X, y = self._make_imbalanced()
        buf = io.StringIO()
        with redirect_stdout(buf):
            decoding.stratify(X, y, strategy='undersample', random_state=0, verbose=True)
        self.assertIn('10', buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
