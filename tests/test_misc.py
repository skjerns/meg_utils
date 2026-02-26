#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for meg_utils.misc — focusing on to_long_df / long_df_to_array.
"""
import sys; sys.path.append('../..')

import numpy as np
import pandas as pd
import pytest

from meg_utils.misc import to_long_df, long_df_to_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# long_df_to_array
# ---------------------------------------------------------------------------

class TestLongDfToArray:

    def test_roundtrip_basic(self):
        """Array → long df → array reproduces the original."""
        arr = RNG.random((4, 5, 3))
        df = to_long_df(arr, columns=['trial', 'time', 'class'], value_name='prob')
        out = long_df_to_array(df, columns=['trial', 'time', 'class'], value_name='prob')
        assert out.shape == arr.shape
        assert np.allclose(out, arr)

    def test_roundtrip_custom_labels(self):
        """Round-trip with non-default axis labels (e.g. real timepoints)."""
        arr = RNG.random((16, 50, 10))
        timepoints = np.arange(-100, 400, 10)  # 50 values
        df = to_long_df(arr, columns=['trial', 'timepoint', 'class'],
                        value_name='probability', timepoint=timepoints)
        out = long_df_to_array(df, columns=['trial', 'timepoint', 'class'],
                               value_name='probability')
        assert out.shape == arr.shape
        assert np.allclose(out, arr)

    def test_roundtrip_string_labels(self):
        """Round-trip when one dimension uses string labels."""
        arr = RNG.random((3, 4))
        labels = ['cat', 'dog', 'bird']
        df = to_long_df(arr, columns=['animal', 'feature'],
                        value_name='score', animal=labels)
        out = long_df_to_array(df, columns=['animal', 'feature'], value_name='score')
        # string labels are sorted lexicographically, so axis-0 order is
        # ['bird', 'cat', 'dog'] — different from input order;
        # just verify shape and that all values are present
        assert out.shape == (3, 4)
        assert np.allclose(np.sort(out.ravel()), np.sort(arr.ravel()))

    def test_shape_only(self):
        """Output shape matches unique counts per column."""
        arr = RNG.random((2, 7, 5))
        df = to_long_df(arr, columns=['a', 'b', 'c'], value_name='v')
        out = long_df_to_array(df, columns=['a', 'b', 'c'], value_name='v')
        assert out.shape == (2, 7, 5)

    def test_1d(self):
        """Works for a 1-D array."""
        arr = RNG.random(10)
        df = to_long_df(arr, columns=['x'], value_name='val')
        out = long_df_to_array(df, columns=['x'], value_name='val')
        assert out.shape == (10,)
        assert np.allclose(out, arr)

    def test_2d(self):
        """Works for a plain 2-D array."""
        arr = RNG.random((8, 6))
        df = to_long_df(arr, columns=['row', 'col'], value_name='v')
        out = long_df_to_array(df, columns=['row', 'col'], value_name='v')
        assert out.shape == (8, 6)
        assert np.allclose(out, arr)

    def test_4d(self):
        """Works for a 4-D array."""
        arr = RNG.random((2, 3, 4, 5))
        df = to_long_df(arr, columns=['a', 'b', 'c', 'd'], value_name='v')
        out = long_df_to_array(df, columns=['a', 'b', 'c', 'd'], value_name='v')
        assert out.shape == (2, 3, 4, 5)
        assert np.allclose(out, arr)

    def test_fill_value_default_nan(self):
        """Missing combinations are filled with NaN by default."""
        arr = RNG.random((3, 3))
        df = to_long_df(arr, columns=['r', 'c'], value_name='v')
        # Drop one row so one combination is absent
        df_partial = df.iloc[:-1].copy()
        out = long_df_to_array(df_partial, columns=['r', 'c'], value_name='v')
        assert np.sum(np.isnan(out)) == 1

    def test_fill_value_custom(self):
        """Missing combinations are filled with the given fill_value."""
        arr = RNG.random((3, 3))
        df = to_long_df(arr, columns=['r', 'c'], value_name='v')
        df_partial = df.iloc[:-1].copy()
        out = long_df_to_array(df_partial, columns=['r', 'c'], value_name='v',
                               fill_value=-1.0)
        assert np.sum(out == -1.0) == 1

    def test_integer_values(self):
        """Integer value columns are preserved (output dtype is float due to fill_value=nan)."""
        arr = np.arange(12).reshape(3, 4)
        df = to_long_df(arr, columns=['r', 'c'], value_name='v')
        out = long_df_to_array(df, columns=['r', 'c'], value_name='v')
        assert out.shape == (3, 4)
        assert np.allclose(out, arr)

    def test_negative_and_float_labels(self):
        """Axis labels with negative floats (e.g. pre-stimulus timepoints)."""
        arr = RNG.random((5, 4))
        times = np.array([-200., -100., 0., 100., 200.])
        df = to_long_df(arr, columns=['time', 'class'], value_name='p',
                        time=times)
        out = long_df_to_array(df, columns=['time', 'class'], value_name='p')
        assert out.shape == (5, 4)
        assert np.allclose(out, arr)

    def test_value_name_forwarded(self):
        """Custom value_name is respected."""
        arr = RNG.random((3, 3))
        df = to_long_df(arr, columns=['x', 'y'], value_name='accuracy')
        out = long_df_to_array(df, columns=['x', 'y'], value_name='accuracy')
        assert np.allclose(out, arr)

    def test_partial_df_shape(self):
        """Shape is still determined by unique values even when rows are missing."""
        arr = np.ones((4, 4))
        df = to_long_df(arr, columns=['r', 'c'], value_name='v')
        # Keep only rows where r != c (remove diagonal)
        df_no_diag = df[df['r'] != df['c']].copy()
        out = long_df_to_array(df_no_diag, columns=['r', 'c'], value_name='v',
                               fill_value=0.0)
        assert out.shape == (4, 4)
        assert np.sum(out == 0.0) == 4   # 4 diagonal entries filled
        assert np.sum(out == 1.0) == 12  # remaining entries


class TestLongDfToArraySubsetColumns:
    """DataFrame has more columns than needed; only the requested ones are used."""

    def test_extra_metadata_column_ignored(self):
        """Extra columns in the df that are not in `columns` are silently ignored."""
        arr = RNG.random((3, 4))
        df = to_long_df(arr, columns=['trial', 'time'], value_name='v')
        df['subject'] = 'sub-01'   # extra column
        df['run'] = 99             # another extra column
        out = long_df_to_array(df, columns=['trial', 'time'], value_name='v')
        assert out.shape == (3, 4)
        assert np.allclose(out, arr)

    def test_subset_of_dim_columns_collapses_last_value(self):
        """Requesting fewer dims than in the df collapses the dropped dim (last write wins)."""
        arr = np.zeros((2, 3, 4))
        arr[0, :, :] = 1.0
        arr[1, :, :] = 2.0
        df = to_long_df(arr, columns=['subject', 'trial', 'time'], value_name='v')
        # Ask only for subject × trial — each (subject, trial) pair has 4 time entries;
        # the final array should still have shape (2, 3) with all entries set.
        out = long_df_to_array(df, columns=['subject', 'trial'], value_name='v')
        assert out.shape == (2, 3)
        # All writes for subject 0 use value 1.0, subject 1 use value 2.0
        assert np.all(out[0, :] == 1.0)
        assert np.all(out[1, :] == 2.0)

    def test_single_column_from_multidim_df(self):
        """Requesting a single column from a 3-D df gives a 1-D array."""
        arr = RNG.random((5, 3, 2))
        df = to_long_df(arr, columns=['a', 'b', 'c'], value_name='v')
        # Only reconstruct along dimension 'a' (5 unique values)
        out = long_df_to_array(df, columns=['a'], value_name='v')
        assert out.shape == (5,)

    def test_reordered_subset(self):
        """Columns can be requested in a different order than they appear in the df."""
        arr = RNG.random((3, 4))
        df = to_long_df(arr, columns=['row', 'col'], value_name='v')
        # Request col before row → transposed result
        out = long_df_to_array(df, columns=['col', 'row'], value_name='v')
        assert out.shape == (4, 3)
        assert np.allclose(out, arr.T)


class TestLongDfToArrayFailures:
    """long_df_to_array should raise meaningful errors on bad input."""

    def test_missing_dimension_column(self):
        """KeyError when a requested column does not exist in the DataFrame."""
        df = pd.DataFrame({'a': [0, 1], 'v': [0.1, 0.2]})
        with pytest.raises(KeyError):
            long_df_to_array(df, columns=['a', 'nonexistent'], value_name='v')

    def test_missing_value_column(self):
        """KeyError when value_name does not exist in the DataFrame."""
        df = pd.DataFrame({'a': [0, 1], 'b': [0, 1], 'v': [0.1, 0.2]})
        with pytest.raises(KeyError):
            long_df_to_array(df, columns=['a', 'b'], value_name='no_such_col')

    def test_empty_columns_list(self):
        """Passing an empty columns list raises an error (0-D arrays are unsupported)."""
        df = pd.DataFrame({'v': [1.0, 2.0]})
        with pytest.raises(Exception):
            long_df_to_array(df, columns=[], value_name='v')
