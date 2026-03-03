#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:23:48 2025

@author: simon
"""

import sys; sys.path.append('../..')

import unittest
import numpy as np
import matplotlib.pyplot as plt
from meg_utils.sigproc import curves, fit_curve
from meg_utils.sigproc import notch
from meg_utils.sigproc import sliding_window
from scipy.fft import rfft, rfftfreq

class TestFitCurve(unittest.TestCase):
    def test_random_sine_truncated(self):
        return
        rng = np.random.default_rng(42)

        for _ in range(15):
            # --- ground-truth pulse ------------------------------------
            freq   = 10.0             # Hz
            amp    = 1.0
            base   = 0.0
            true_t = np.linspace(0, 0.35, 1000)
            loc  = rng.uniform(min(true_t),
                                 max(true_t) - max(true_t) / freq)

            truth = curves.sine_truncated(
                true_t,
                frequency=freq,
                amplitude=amp,
                loc=loc,
                baseline=base,
            )

            # --- down-sample to faux “experiment” ---------------------
            data_sfreq = rng.uniform(15, 40)     # random low sfreq
            samp_t     = np.arange(0, true_t[-1], 1 / data_sfreq)
            samp_y     = np.interp(samp_t, true_t, truth)

            # -----------------------------------------------------------------
            # DATA-DRIVEN INITIAL GUESSES
            # -----------------------------------------------------------------
            amp_guess   = samp_y.max()             # peak height
            shift_guess = samp_t[np.argmax(samp_y)]  # peak location

            spec = {
                "frequency": [(freq, freq), freq],                 # fixed at 10 Hz
                "amplitude": [(amp_guess, 2 * amp_guess), amp_guess],      # start at peak
                "loc":     [(0, max(true_t) - max(true_t) / freq), shift_guess],
                "baseline":  [(0, 0), 0.0],                        # pinned to 0
            }

            # --- fit -------------------------------------------------
            plt.figure()
            plt.plot(true_t, truth, label='truth', c='gray')
            _, _, est = fit_curve(
                samp_y,
                data_sfreq=data_sfreq,
                model=curves.sine_truncated,
                curve_sfreq=1000,
                curve_params=spec,
                plot_fit=True,
            )

            # --- assertions -----------------------------------------
            self.assertAlmostEqual(est["loc"],     loc, delta=0.01)
            self.assertAlmostEqual(est["amplitude"], amp,   delta=0.05)
            self.assertAlmostEqual(est["frequency"], freq,  delta=0.1)
            self.assertAlmostEqual(est["baseline"],  base,  delta=0.02)



# ------------------------------------------------------------------ helpers
def _psd(x, sfreq):
    """Power spectral density on last axis."""
    return np.abs(rfft(x, axis=-1)) ** 2, rfftfreq(x.shape[-1], 1 / sfreq)

# ------------------------------------------------------------------ tests
class FilterTests(unittest.TestCase):


    def test_notch(self):
        sfreq = 1000
        shapes = [(3, 2000), (1, 3, 2000)]
        t = np.arange(2000) / sfreq
        rng = np.random.default_rng(7)
        tone = np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(t.size)

        for shape in shapes:
            data = np.broadcast_to(tone, shape).copy()
            original = data.copy()

            out = notch(data, freqs=[50], sfreq=sfreq)


class TestSlidingWindow(unittest.TestCase):

    def test_output_shape_1d(self):
        """Shape is (n_windows, win_size) for a 1-D input."""
        out = sliding_window(np.arange(10), win_size=4, stride=1)
        self.assertEqual(out.shape, (7, 4))

    def test_output_shape_2d(self):
        """Shape along the slid axis becomes n_windows, win_size is appended."""
        out = sliding_window(np.ones((3, 20)), win_size=5, stride=2)
        self.assertEqual(out.shape, (3, 8, 5))

    def test_output_shape_axis0(self):
        """Sliding along axis=0 replaces the first dimension with n_windows."""
        out = sliding_window(np.ones((20, 3)), win_size=5, stride=2, axis=0)
        self.assertEqual(out.shape, (8, 3, 5))

    def test_output_shape_3d(self):
        """Correct shape for a 3-D input sliding along the middle axis."""
        out = sliding_window(np.ones((2, 10, 4)), win_size=3, stride=2, axis=1)
        self.assertEqual(out.shape, (2, 4, 4, 3))

    def test_window_values(self):
        """Each window contains the correct consecutive elements."""
        out = sliding_window(np.arange(6), win_size=3, stride=1)
        np.testing.assert_array_equal(out[0], [0, 1, 2])
        np.testing.assert_array_equal(out[1], [1, 2, 3])
        np.testing.assert_array_equal(out[2], [2, 3, 4])
        np.testing.assert_array_equal(out[3], [3, 4, 5])

    def test_stride_skips_correctly(self):
        """stride > 1 advances the window start by that many elements."""
        out = sliding_window(np.arange(10), win_size=3, stride=3)
        np.testing.assert_array_equal(out[0], [0, 1, 2])
        np.testing.assert_array_equal(out[1], [3, 4, 5])
        np.testing.assert_array_equal(out[2], [6, 7, 8])

    def test_is_view_shares_memory(self):
        """Output shares memory with the input — no data copy is made."""
        arr = np.arange(100, dtype=float)
        out = sliding_window(arr, win_size=10, stride=1)
        self.assertTrue(np.shares_memory(arr, out))

    def test_view_reflects_source_changes(self):
        """Mutating the source array is immediately visible through the view."""
        arr = np.arange(20, dtype=float)
        out = sliding_window(arr, win_size=3, stride=1)
        arr[0] = 999.0
        self.assertEqual(out[0, 0], 999.0)

    def test_read_only(self):
        """Output is read-only; writing to it raises ValueError."""
        out = sliding_window(np.arange(10, dtype=float), win_size=3, stride=1)
        with self.assertRaises(ValueError):
            out[0, 0] = 99.0

    def test_win_size_equals_length(self):
        """win_size equal to axis length returns exactly one window."""
        arr = np.arange(5)
        out = sliding_window(arr, win_size=5, stride=1)
        self.assertEqual(out.shape, (1, 5))
        np.testing.assert_array_equal(out[0], arr)

    def test_win_size_exceeds_length_raises(self):
        """win_size larger than the axis length raises ValueError."""
        with self.assertRaises(ValueError):
            sliding_window(np.arange(5), win_size=6)

    def test_invalid_stride_raises(self):
        """stride <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            sliding_window(np.arange(10), win_size=3, stride=0)

    def test_invalid_win_size_raises(self):
        """win_size <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            sliding_window(np.arange(10), win_size=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
