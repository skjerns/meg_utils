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

def resample(array, o_sfreq, t_sfreq, n_jobs=-1, verbose=False):
    """
    resample a signal using MNE resample functions
    This automatically is optimized for EEG applying filters etc

    1D : (timestep)
    2D : (channels, timestep)
    3D : (epochs, channels, timestep)

    :param array:     a 1D/2D/3D data array
    :param o_sfreq: the original sampling frequency
    :param t_sfreq: the target sampling frequency
    :returns: the resampled signal
    """
    if o_sfreq==t_sfreq: return array
    array = np.atleast_3d(array)

    if array.ndim>3:
        raise ValueError(f'Too many dimensions in array: {array.ndim}')

    ch_names=['ch{}'.format(i) for i in range(array.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=o_sfreq, ch_types=['eeg']*array.shape[1])
    raw_mne = mne.EpochsArray(array, info, tmin=0, verbose=verbose)

    resampled = raw_mne.resample(t_sfreq, n_jobs=n_jobs, verbose=verbose)
    new_raw = resampled.get_data().squeeze()
    return new_raw.astype(array.dtype, copy=False)


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


def estimate_paf(
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


def get_alpha_peak(raw, alpha_bounds=[7, 14], return_spectrum=False,
                   plot_spectrum=False):
    """
    retrieve the peak frequency of the alpha activity

    Parameters
    ----------
    raw : TYPE
        the mne Raw file to examine.
    alpha_bounds : list
        lower and upper bounds of alpha band search window
    return_spectrum : bool, optional
        if True return the spectrum and frequencies

    Returns
    -------
    alpha_peak : float
        peak of the alpha band.
    source : int
        sensor with highest power in that band
    """
    assert len(alpha_bounds)==2
    alpha_l, alpha_u = alpha_bounds
    picks = mne.read_vectorview_selection('occipital', info=raw.info)
    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']
    freqs, w = welch(data, fs=sfreq, nperseg=sfreq*10, scaling='spectrum')
    idx_alpha = (alpha_l<freqs) & (freqs<alpha_u)

    # get sensor with highest alpha power
    idx_source = np.argmax(w.mean(1))
    source_sensor = picks[idx_source]

    # get power peak
    w_alpha = detrend(w[:, idx_alpha].mean(0))
    alpha_peak = freqs[idx_alpha][np.argmax(w_alpha)]

    if freqs[np.argmax(alpha_bounds[0]<freqs)]==alpha_peak:
        warnings.warn(f'alpha peak lays at bounds, not likely? {alpha_peak=}')
    if freqs[np.argmax(alpha_bounds[1]<freqs)]==alpha_peak:
        warnings.warn(f'alpha peak lays at bounds, not likely? {alpha_peak=}')

    if plot_spectrum:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        fig, ax = plt.subplots(1, 1)
        ax.plot(freqs[idx_alpha], w_alpha)
        inset_ax = inset_axes(ax, width="20%", height="20%", loc="upper right")  # Adjust size and location

    if return_spectrum:
        alpha_peak, source_sensor, freqs, w

    return alpha_peak, source_sensor


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
