#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 12:24:31 2025

various MEG signal processing functions

@author: simon.kern
"""
import warnings
import mne
import numpy as np
from collections import namedtuple
from scipy.signal import welch
from scipy.signal import detrend
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import numpy as np
import mne

def resample(array, o_sfreq, t_sfreq, n_jobs=-1, verbose=False):
    """
    Resample EEG data using MNE.

    Parameters:
        array (ndarray): 1D, 2D, or 3D input (time | channels x time | epochs x channels x time)
        o_sfreq (float): Original sampling frequency
        t_sfreq (float): Target sampling frequency
        n_jobs (int): Number of parallel jobs (default: -1)
        verbose (bool): MNE verbosity flag

    Returns:
        ndarray: Resampled data with same shape as input (squeezed if needed)
    """
    if o_sfreq == t_sfreq:
        return array
    array = np.atleast_3d(array)
    if array.ndim > 3:
        raise ValueError(f'Too many dimensions in array: {array.ndim}')
    ch_names = ['ch{}'.format(i) for i in range(array.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=o_sfreq, ch_types=['eeg'] * array.shape[1])
    raw_mne = mne.EpochsArray(array, info, tmin=0, verbose=verbose)
    resampled = raw_mne.resample(t_sfreq, n_jobs=n_jobs, verbose=verbose)
    return resampled.get_data().squeeze().astype(array.dtype, copy=False)

def bandpass(data, lfreq=None, ufreq=None, sfreq=100, verbose=False, **kwargs):
    """
    Apply bandpass filter using MNE.

    Parameters:
        data (ndarray): 2D or 3D input (channels x time | epochs x channels x time)
        lfreq (float): Low cutoff frequency
        ufreq (float): High cutoff frequency
        sfreq (float): Sampling rate
        verbose (bool): MNE verbosity flag
        **kwargs: Additional arguments to MNE filter

    Returns:
        ndarray: Bandpass-filtered data (squeezed)
    """
    if data.ndim > 3 or data.ndim < 2:
        raise ValueError(f'Invalid data dimensions for bandpass: {data.ndim}')
    data = np.atleast_3d(data)
    ch_names = ['ch{}'.format(i) for i in range(data.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * data.shape[1])
    raw = mne.EpochsArray(data, info, tmin=0, verbose=verbose)
    raw.filter(l_freq=lfreq, h_freq=ufreq, verbose=verbose, **kwargs)
    return raw.get_data().squeeze().astype(data.dtype, copy=False)



def sliding_window(array, win_size, stride=1, axis=-1):
    """Return a read-only sliding window view with window dimension last.

    Args:
        array: Input ndarray.
        win_size: Size of the rolling window along the specified axis.
        stride: Step between window starts.
        axis: Axis along which to slide.

    Returns:
        ndarray: Read-only strided view whose shape equals the input shape
            with the length along `axis` replaced by `n_windows`, and an
            extra trailing dimension of size `win_size`.
    """
    arr = np.asarray(array)
    if win_size <= 0 or stride <= 0:
        raise ValueError("win_size and stride must be positive")

    axis = np.core.numeric.normalize_axis_index(axis, arr.ndim)
    L = arr.shape[axis]
    if win_size > L:
        raise ValueError("win_size cannot exceed axis length")

    n_windows = 1 + (L - win_size) // stride

    shape = list(arr.shape)
    shape[axis] = n_windows
    shape.append(win_size)

    base_stride = arr.strides[axis]
    strides = list(arr.strides)
    strides[axis] = base_stride * stride
    strides.append(base_stride)

    view = np.lib.stride_tricks.as_strided(arr, shape=tuple(shape), strides=tuple(strides))
    view.setflags(write=False)
    return view


def notch(data, freqs=None, notch_widths=None, sfreq=100, verbose=False,
          n_jobs=-1, **kwargs):
    """
    Apply notch filter using MNE RawArray.

    Parameters:
        data (ndarray): EEG data. 2D (n_channels x n_times) or 3D (n_epochs x n_channels x n_times)
        freqs (list or float): Frequencies to filter out (e.g., line noise)
        sfreq (float): Sampling rate
        verbose (bool): MNE verbosity flag
        n_jobs (int): Number of parallel jobs (default: -1)
        **kwargs: Extra keyword arguments passed to MNE notch_filter

    Returns:
        ndarray: Filtered data, same shape as input (squeezed if needed)
    """
    if freqs is None:
        freqs = [50, 100]

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(f"Unsupported data shape {data.shape}, must be 2D or 3D")

    n_epochs, n_channels, n_times = data.shape
    ch_names = [f'ch{i}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)

    def filter_epoch(epoch):
        raw = mne.io.RawArray(epoch, info, copy='both', verbose=verbose)
        raw.notch_filter(freqs=freqs, notch_widths=notch_widths, verbose=verbose, **kwargs)
        return raw.get_data()

    filtered = Parallel(n_jobs=n_jobs)(
        delayed(filter_epoch)(data[i]) for i in range(n_epochs)
    )

    return np.stack(filtered).astype(data.dtype).squeeze()




def get_ch_neighbours(ch_name, n=9, return_idx=False, plot=False):
    """retrieve the n neighbours of a given electrode location.
    Count includes the given origin electrode location"""
    layout = mne.channels.read_layout('Vectorview-all')
    positions = {name.replace(' ', ''):pos[:3] for name, pos in zip(layout.names, layout.pos, strict=True)}

    Point = namedtuple('Point', 'name x y z')
    ch = Point(ch_name, *positions[ch_name])
    chs = [Point(ch, *pos) for ch, pos in positions.items()]
    chs = [ch for ch in chs if not (('EOG' in ch.name) or ('IO' in ch.name))]

    dist = lambda p: (p.x - ch.x)**2 + (p.y - ch.y)**2 + (p.z - ch.z)**2

    chs_sorted = sorted(chs, key=dist)

    chs_out = [ch.name for ch in chs_sorted[:n]]

    ch_as_in_raw = sorted([ch.replace(' ', '') for ch in layout.names])

    if plot:
        layout.plot(picks=[list(positions).index(ch) for ch in chs_out])
    return sorted([ch_as_in_raw.index(ch) for ch in chs_out]) if return_idx else chs_out




def estimate_peak_alpha_freq(
    raw_or_fname,
    *,
    picks=None,
    alpha_band=(7.0, 13.0),
    spec_range=(2.0, 30.0),
    bandwidth=2.0,
    notch=(50,),
    hp=1.0,
    lp=40.0,
    use_specparam=True,
    specparam_kwargs=None,
    method="peak",          # {"peak", "cog"}
    plot=False,
    verbose=True,
):
    """
    Estimate peak-alpha frequency (PAF) from an MEG recording.

    Parameters
    ----------
    raw_or_fname : mne.io.BaseRaw | str | pathlib.Path
        Pre-loaded Raw object or a filename that MNE can read.
    picks : list | None
        Channel names / indices.  Default → all MEG gradiometers.
    alpha_band : tuple(float, float)
        Frequency window inside which to look for the alpha peak.
    spec_range : tuple(float, float)
        Full spectrum range that is estimated & modelled.
    bandwidth : float
        Slepian multitaper smoothing ±Hz.
    notch : tuple | None
        Frequencies for notch filtering, e.g. (50, 100).  None → skip.
    hp, lp : float | None
        High-/low-pass FIR filter cut-offs (Hz).  None → skip either.
    use_specparam : bool
        If True, subtract the 1/f background with specparam.
    specparam_kwargs : dict | None
        Extra kwargs passed to specparam.SpectralModel.
    method : {"peak", "cog"}
        * peak – highest point inside alpha_band
        * cog  – power-weighted centre of gravity
    plot : bool
        Plot PSD & PAF if True.
    verbose : bool
        Control console output.

    Returns
    -------
    dict
        { "paf", "cog", "freqs", "spectrum", "spectrum_osc", "model" }
    """
    # ------------------------------------------------------------------
    # 0. Load raw
    # ------------------------------------------------------------------
    if isinstance(raw_or_fname, mne.io.BaseRaw):
        raw = raw_or_fname.copy().load_data()
    else:
        raw = mne.io.read_raw(raw_or_fname, preload=True, verbose=verbose)

    # ------------------------------------------------------------------
    # 1. Quick cleaning
    # ------------------------------------------------------------------

    if hp or lp:
        raw.filter(l_freq=hp, h_freq=lp, fir_design="firwin", verbose=verbose,
                   n_jobs=-1)

    # ------------------------------------------------------------------
    # 2. Pick sensors
    # ------------------------------------------------------------------
    if picks is None:
        picks = mne.pick_types(raw.info, meg="grad")

    # ------------------------------------------------------------------
    # 3. PSD (multitaper)
    # ------------------------------------------------------------------
    fmin, fmax = spec_range
    psd = raw.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        verbose=verbose,
    )
    freqs = psd.freqs

    spectrum = 10 * psd.get_data().mean(0)       # dB, averaged sensors & time
    spectrum_osc = spectrum.copy()            # will be flattened later
    model = None

    # ------------------------------------------------------------------
    # 4. Remove 1/f background with specparam (optional)
    # ------------------------------------------------------------------
    if use_specparam:
        try:
            from specparam import SpectralModel
        except ImportError as e:
            raise RuntimeError("Install specparam:  pip install specparam") from e

        kwargs = dict(
            peak_width_limits=[0.5, 4],
            min_peak_height=0.1,
            max_n_peaks=4,
            peak_threshold=2.0,
            verbose=verbose,
        )
        if specparam_kwargs:
            kwargs.update(specparam_kwargs)

        model = SpectralModel(**kwargs)
        model.fit(freqs, spectrum, spec_range)
        # Try the two private attributes used across specparam versions
        bg = getattr(model, "_ap_fit", None)
        if bg is None:
            bg = getattr(model, "_bg_fit", np.zeros_like(spectrum))
        spectrum_osc = spectrum - bg

    # ------------------------------------------------------------------
    # 5. Extract peak / CoG
    # ------------------------------------------------------------------
    mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    spec_a = spectrum_osc[mask]
    freqs_a = freqs[mask]

    cog = float(np.sum(freqs_a * spec_a) / np.sum(spec_a))
    paf = float(cog if method == "cog" else freqs_a[np.argmax(spec_a)])

    # ------------------------------------------------------------------
    # 6. Plot (optional)
    # ------------------------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(freqs, spectrum, label="PSD (dB)")
        if use_specparam:
            plt.plot(freqs, bg, "--", label="1/f fit")
            plt.plot(freqs, spectrum_osc, label="PSD − 1/f")
        plt.axvline(paf, color="r", lw=1.5, label=f"PAF = {paf:.2f} Hz")
        plt.xlim(fmin, fmax)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title("Peak-Alpha Frequency")
        plt.legend()
        plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Return results
    # ------------------------------------------------------------------
    return dict(
        paf=paf,
        cog=cog,
        freqs=freqs,
        spectrum=spectrum,
        spectrum_osc=spectrum_osc,
        model=model,
    )

import warnings
import numpy as np
import mne
from scipy.signal import welch, detrend


def get_alpha_peak(raw_or_data,
                   sfreq: float | None = None,
                   alpha_bounds: tuple[float, float] = (7, 14),
                   return_spectrum: bool = False,
                   plot_spectrum: bool = False):
    """
    Alpha-peak finder for MNE Raw, (data, sfreq) tuples, or plain ndarrays.

    Returns
    -------
    alpha_peak : float
        Peak frequency in the α-band
    src_chan   : int
        Index of the channel with the highest α-band power
    freqs, psd : if `return_spectrum` is True
    """
    lo, hi = alpha_bounds

    # ------------------------------------------------------------------ #
    # Unwrap input                                                       #
    # ------------------------------------------------------------------ #
    if isinstance(raw_or_data, mne.io.BaseRaw):          # MNE Raw object
        if sfreq is not None:
            raise ValueError("Provide either an MNE-Raw or (data, sfreq), "
                             "not both.")
        picks = mne.read_vectorview_selection('occipital',
                                              info=raw_or_data.info)
        data = raw_or_data.get_data(picks=picks)         # shape: ch × time
        sfreq = raw_or_data.info['sfreq']
    else:                                                # ndarray input
        if sfreq is None:
            raise ValueError("When passing an ndarray you must supply sfreq.")
        data = np.asarray(raw_or_data)
        picks = np.arange(data.shape[0])                 # all channels

    # Ensure shape is (n_channels, n_times) ----------------------------- #
    if data.ndim == 1:                                   # single channel
        data = data[np.newaxis, :]

    assert data.ndim <3 , 'only 1d or 2d allowed'

    if data.shape[0] > data.shape[1]:
        raise ValueError("Expected shape (n_channels, n_times); "
                         f"got {data.shape}. Swap axes if needed.")

    # ------------------------------------------------------------------ #
    # Power spectral density (Welch)                                     #
    # ------------------------------------------------------------------ #
    # Compute along time axis (axis = 1).  Result: psd shape =
    # (n_channels, n_freqs)
    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=int(sfreq * 5),
        scaling='spectrum',
        axis=1,                     # ← critical change
    )

    sel = (freqs > lo) & (freqs < hi)

    # Detrend before peak pick to avoid broad slopes
    alpha_psd = detrend(psd[:, sel].mean(0))
    alpha_peak = float(freqs[sel][np.argmax(alpha_psd)])

    if (alpha_peak-lo)<=0.25 or (hi-alpha_peak)<=0.25 :
        warnings.warn(f"α-peak lies on the band edge: {alpha_peak:.2f} Hz")

    # ------------------------------------------------------------------ #
    # Optional plot                                                      #
    # ------------------------------------------------------------------ #
    if plot_spectrum:
        import matplotlib.pyplot as plt
        plt.plot(freqs[sel], alpha_psd)
        plt.axvline(alpha_peak, ls='--')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.show()

    if return_spectrum:
        return alpha_peak, freqs, psd
    return alpha_peak








def get_alpha_phase(raw, alpha_peak, bandwidth=1.5):
    raw_grad = raw.copy().pick('grad')
    sfreq = raw_grad.info['sfreq']
    raw_grad.filter(alpha_peak-bandwidth, alpha_peak+bandwidth, n_jobs=-1,
                    fir_design='firwin', phase='zero-double')
    psd = raw_grad.apply_hilbert(n_jobs=-1)

    # get instantaneous phase and amplitude
    phases = np.angle(psd.get_data())
    amplitude = np.abs(psd.get_data())

    # take electrode with highest mean amplitude, i.e. strongest alpha power
    ch_alpha_source = raw_grad.ch_names[np.argmax(amplitude.mean(1))]

    t_peak_alpha = np.argmax(np.convolve(amplitude.mean(0), np.ones(int(sfreq))/sfreq))
    phases_peak = phases[:, t_peak_alpha]
    amplitude_peak = amplitude[:, t_peak_alpha]
    return ch_alpha_source, phases_peak, amplitude_peak


def create_oscillation(hz, sfreq=100, n_samples=None, n_seconds=None,
                       phi_rad=None, phi_deg=None, amp=1.0):
    """
    Generate a sinusoidal waveform with amplitude modulation per cycle.

    Parameters
    ----------
    hz : float
        Frequency of the sine wave in Hertz.
    sfreq : float, default 100
        Sampling rate in samples/second.
    n_samples : int, optional
        Number of samples to return. Mutually exclusive with `n_seconds`.
    n_seconds : float, optional
        Length of the waveform in seconds. Mutually exclusive with `n_samples`.
    phi_rad : float, optional
        Initial phase offset in radians [0–2π]. Mutually exclusive with `phi_deg`.
    phi_deg : float, optional
        Initial phase offset in degrees [0–360]. Mutually exclusive with `phi_rad`.
    amp : float or array-like, default 1.0
        If float, scales entire signal. If array-like, each cycle is scaled by its element;
        wraps if shorter than total cycles.

    Returns
    -------
    numpy.ndarray
        1-D array of shape (`n_samples`,) containing the modulated sine wave.
    """
    # phase handling
    if phi_rad is not None and phi_deg is not None:
        raise ValueError("Specify only one of `phi_rad` or `phi_deg`.")
    phase = phi_rad if phi_rad is not None else (
        np.deg2rad(phi_deg) if phi_deg is not None else 0.0)

    # duration handling
    if (n_samples is None) == (n_seconds is None):
        raise ValueError("Specify exactly one of `n_samples` or `n_seconds`.")
    if n_samples is None:
        n_samples = int(round(n_seconds * sfreq))

    # time vector and base waveform
    t = np.arange(n_samples, dtype=float) / sfreq
    sine = np.sin(2 * np.pi * hz * t + phase)

    # amplitude modulation
    if np.isscalar(amp):
        return amp * sine

    amp_arr = np.asarray(amp, dtype=float)
    # determine cycle index for each sample
    cycle_idx = (np.floor(hz * t).astype(int) % amp_arr.size)
    return sine * amp_arr[cycle_idx]


def wave_speed_cm(
        phases: np.ndarray,
        idx_source: int,
        pos: np.ndarray,
        freq_hz = None,
    ) -> float:
    """
    Estimate the phase-velocity (cm s⁻¹) of a travelling cortical wave
    from MEG phase angles.

    Parameters
    ----------
    phases      : (n_sensors,) array_like
        Instantaneous phase (radians) for every sensor at one time-point.
    idx_source  : int
        Index of the sensor taken as the wave’s point of origin.
    pos_3d      : (n_sensors, 3) array_like
        Sensor positions in 3-D head coordinates (metres *or* the same
        linear unit you want the result to be expressed in).
        Used when geometry="3d".
    pos_2d      : (n_sensors, 2) array_like, optional
        Flattened 2-D sensor positions. Required when geometry="2d".
    freq_hz     : float, optional
        Carrier frequency of the oscillation (Hz).
        If None, the *phase-gradient speed* (cm cycle⁻¹) is returned:
            v = 1 / |∇φ|   (distance that corresponds to 2π phase shift)
    geometry    : {"3d", "2d"}, default "3d"
        Distance metric:
          • "3d" – geodesic distance on the best-fitting sphere.
          • "2d" – Euclidean distance in the supplied 2-D map.

    Returns
    -------
    speed_cm_s  : float
        Wave speed in centimetres s⁻¹ (if `freq_hz` given) or
        centimetres cycle⁻¹ (otherwise).
    """
    phases = np.asarray(phases, float)
    dphi = np.angle(np.exp(1j * (phases - phases[idx_source])))
    # ------------------------------------------------------------------
    # 1. Distance from source
    # ------------------------------------------------------------------
    if pos.shape[1] == 3:
        p = np.asarray(pos, float)
        # radius of best-fitting sphere (average sensor norm)
        R = np.mean(np.linalg.norm(p, axis=1))
        # unit vectors
        p_unit   = p / np.linalg.norm(p, axis=1, keepdims=True)
        p0_unit  = p_unit[idx_source]
        # great-circle (geodesic) angle to source
        theta = np.arccos(np.clip(p_unit @ p0_unit, -1.0, 1.0))
        dist = R * theta                                     # in metres
    elif pos.shape[1] == 2:
        q = np.asarray(pos, float)
        dist = np.linalg.norm(q - q[idx_source], axis=1)     # in map units
    else:
        raise ValueError("pos must be 3d or 2d but is {pos.ndim=}")

    idx = np.argsort(dist)         # increasing distance from the source
    dphi_unw = dphi.copy()

    # sort the angle vector according to distance
    for i_prev, i_curr in zip(idx[:-1], idx[1:]):
        jump = dphi_unw[i_curr] - dphi_unw[i_prev]
        if   jump >  np.pi: dphi_unw[i_curr:] -= 2*np.pi
        elif jump < -np.pi: dphi_unw[i_curr:] += 2*np.pi
    # ------------------------------------------------------------------
    # 2. Phase-gradient (k) via least-squares:  dφ ≈ k·d
    # ------------------------------------------------------------------
    # Solve slope k that minimises ||k·dist − dphi||
    k, _ = np.polyfit(dist, dphi_unw, 1)        # k in rad / (unit of `dist`)
    k = np.abs(k)

    # ------------------------------------------------------------------
    # 3. Convert to speed
    #    v = 2πf / |k|            (m s⁻¹)          if f known
    #    v = 2π   / |k|           (m cycle⁻¹)      otherwise
    # ------------------------------------------------------------------
    if freq_hz is None:
        speed_m = (2 * np.pi) / k            # m per cycle
    else:
        speed_m = (2 * np.pi * freq_hz) / k  # m per second

    return float(speed_m * 100.0)            # → cm s⁻¹  or  cm cycle⁻¹


import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


class curves:
    """Collection of curve primitives.  Each function takes `time` first."""

    # --- model functions -------------------------------------------------

    @staticmethod
    def gaussian(time, amplitude, mean, std, baseline):
        return amplitude * norm.pdf(time, mean, std) + baseline

    @staticmethod
    def sine_truncated(
        time,
        *,
        frequency: float,
        amplitude: float,
        loc: float,
        baseline: float = 0.0,
    ):
        """
        Truncated half-sine pulse whose **centre** (peak) is at ``loc``.

        The pulse spans one full period (``1 / frequency``) so that

        * left edge  = ``loc – 1 / (2 · frequency)``
        * right edge = ``loc + 1 / (2 · frequency)``
        * peak       = ``loc``

        Outside that window the function is flat at ``baseline``.

        Parameters
        ----------
        time : array-like
            Times at which to evaluate the waveform.
        frequency : float
            Oscillation frequency in hertz (Hz).
        amplitude : float
            Peak height above ``baseline``.  The function ranges
            from ``baseline`` up to ``baseline + amplitude``.
        loc : float
            Centre (mean) of the pulse, analogous to *loc* in
            :pyfunc:`scipy.stats.norm`.
        baseline : float, default 0
            Constant value returned outside the active window.

        Returns
        -------
        numpy.ndarray
            Waveform evaluated at ``time`` (same shape as input).
        """
        time = np.asarray(time, dtype=float)
        period = 1.0 / frequency
        start = loc - period / 2.0     # left edge of the window
        end   = loc + period / 2.0     # right edge

        # half-sine defined over 0 … π
        phase = 2.0 * np.pi * frequency * (time - start) - 0.5 * np.pi
        y = (amplitude / 2.0) * np.sin(phase) + baseline + (amplitude / 2.0)

        # flatten outside the active window
        y[(time < start) | (time > end)] = baseline
        return y

    # --------------------------------------------------------------------


def fit_curve(data, data_sfreq=1 / 1.25, *,  model=curves.gaussian,
              curve_sfreq=100, curve_params, plot_fit=False):

    """
     Fit a parametric curve to 1-D data with L-BFGS-B.

     Args
     ----
     data : array-like
         1-D signal.
     data_sfreq : float, default 0.8
         Sampling rate of `data` (Hz).
     model : callable, default curves.gaussian
         Function f(t, **params) generating the curve.
     curve_sfreq : float, default 100
         Sampling rate (Hz) of the returned fitted curve.
     curve_params : dict
         Mapping ``{name: ((lo, hi), p0)}`` of bounds and initial guess.
     plot_fit : bool | matplotlib.axes.Axes, optional
         • ``False`` – no plot
         • ``True`` – plot on current axes
         • ``Axes`` – plot on the given axes.

     Returns
     -------
     fine_t : ndarray
         High-resolution time axis (s) at `curve_sfreq`.
     fitted : ndarray
         Model evaluated at `fine_t` with the optimized parameters.
     best : dict
         Optimised parameter values ``{name: value}``.

     Notes
     -----
     Minimises sum-squared error, with bounds enforced. If `plot_fit` is
     truthy, overlays the raw data, initial guess, and final fit.
     """
    data = np.asarray(data, dtype=float)
    t = np.arange(data.size) / data_sfreq

    # --- unpack parameter meta ------------------------------------------
    names  = list(curve_params.keys())
    p0     = [curve_params[n][1] for n in names]          # initial guess
    bounds = [tuple(curve_params[n][0]) for n in names]

    # --- objective -------------------------------------------------------
    def sse(p):
        kwargs = dict(zip(names, p))
        return np.sum((data - model(t, **kwargs)) ** 2)

    # --- optimise -------------------------------------------------------
    res  = minimize(sse, p0, bounds=bounds, method="L-BFGS-B")
    best = dict(zip(names, res.x))

    # --- high-res fitted curve ------------------------------------------
    fine_t = np.arange(0, t[-1] + 1 / curve_sfreq, 1 / curve_sfreq)
    fitted = model(fine_t, **best)

    if plot_fit != False:
        if not isinstance(plot_fit, plt.Axes):
            ax = plt.gca()
        # first plot the initial guess
        fine_t = np.arange(0, t[-1] + 1 / curve_sfreq, 1 / curve_sfreq)
        init   = model(fine_t, **dict(zip(names, p0)))
        ax.plot(fine_t, init, ":", lw=1.2, label="initial guess", c='gray')

        c = ax._get_lines.get_next_color()

        # next the fitted curve
        ax.scatter(t, data, s=20, label="data", c=c)
        ax.plot(fine_t, fitted, "--", lw=1.4, label="fit", c=c)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.figure.tight_layout()

    return fine_t, fitted, best



def interpolate(times, data, n_samples=None, kind='linear', axis=-1):
    """interpolate data sampled at times to evenly spaced
    expected is row-wise interpolation, e.g. 1x10, 2x10

    kind: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear',
    'quadratic', 'cubic', 'previous', or 'next'. 'zero', 'slinear',"""
    from scipy.interpolate import interp1d

    if n_samples is None:
        n_samples = len(np.unique(times))

    if set(times)!=len(times):
        times_unique = np.unique(times)
        mean = [np.mean(data[times==t], 0) for t in times_unique]
        data = np.transpose(mean)
        times = times_unique

    # Define evenly spaced time grid
    even_timepoints = np.linspace(times.min(), times.max(), num=n_samples)

    # Interpolation function (linear, can change to 'cubic', 'quadratic', etc.)
    interp_func = interp1d(times, data, kind=kind)  # or 'cubic'

    # Interpolated values
    even_spaced_values = interp_func(even_timepoints)
    return even_timepoints, even_spaced_values


def polyfit(times, data, n_samples=None, degree=3, axis=-1):
    """Fit a polynomial to data sampled at times and evaluate it at evenly spaced points.

    Parameters:
    - times: 1D array of timepoints (unevenly spaced)
    - data: array of values (can be 1D or 2D)
    - n_samples: number of evenly spaced samples to generate; defaults to len(times)
    - degree: degree of polynomial
    - axis: axis along which to fit (only applies to 2D arrays)

    Returns:
    - fitted values at evenly spaced timepoints
    """
    import numpy as np

    times = np.asarray(times)
    data = np.asarray(data)

    if n_samples is None:
        n_samples = len(np.unique(times))

    even_timepoints = np.linspace(times.min(), times.max(), num=n_samples)

    if data.ndim == 1:
        coeffs = np.polyfit(times, data, deg=degree)
        poly = np.poly1d(coeffs)
        return even_timepoints, poly(even_timepoints)

    elif data.ndim == 2:
        # Move the target axis to be first for iteration
        data = np.moveaxis(data, axis, 0)
        fitted = np.array([
            np.poly1d(np.polyfit(times, d, deg=degree))(even_timepoints)
            for d in data
        ])
        # Move axis back to original
        return even_timepoints, np.moveaxis(fitted, 0, axis)

    else:
        raise ValueError("Only 1D or 2D data is supported.")
