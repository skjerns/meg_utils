#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:23:48 2025

@author: simon
"""

import sys; sys.path.append('../..')
from meg_utils import sigproc


import os
import unittest
import numpy as np
import scipy
from scipy import io
from tqdm import tqdm
from tdlm import plotting
import matplotlib.pyplot as plt
import tdlm
import unittest
from meg_utils.sigproc import curves, fit_curve
from meg_utils.sigproc import bandpass, notch

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


import unittest
import numpy as np
from scipy.fft import rfft, rfftfreq

# ------------------------------------------------------------------ helpers
def _psd(x, sfreq):
    """Power spectral density on last axis."""
    return np.abs(rfft(x, axis=-1)) ** 2, rfftfreq(x.shape[-1], 1 / sfreq)

# ------------------------------------------------------------------ tests
class FilterTests(unittest.TestCase):

    def test_bandpass(self):
        sfreq = 100
        shapes = [(1, 1000), (4, 1000), (2, 4, 1000)]
        combos = [(1, 40), (None, 30), (10, None)]
        rng = np.random.default_rng(42)

        for shape in shapes:
            data = rng.standard_normal(shape)
            original = data.copy()

            for lfreq, ufreq in combos:
                out = bandpass(data, lfreq=lfreq, ufreq=ufreq, sfreq=sfreq)
                ps_orig, freqs = _psd(original.reshape(-1, shape[-1]), sfreq)
                ps_filt, _     = _psd(out.reshape(-1, shape[-1]), sfreq)

                mask = np.ones_like(freqs, bool)
                if lfreq is not None:
                    mask &= freqs < lfreq
                if ufreq is not None:
                    mask &= freqs > ufreq

                self.assertLess(ps_filt[:, mask].mean(),
                                0.5 * ps_orig[:, mask].mean(),
                                msg=f"Bandpass {lfreq}-{ufreq} Hz failed")
            # input must stay intact
            self.assertTrue(np.array_equal(data, original),
                            msg="bandpass modified input in-place")

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


unittest.main(verbosity=2)
