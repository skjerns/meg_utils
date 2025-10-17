# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:13 2024

@author: Simon Kern (@skjerns)
"""
import os
import logging
import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import robust_scale as _robust_scale
from .constants import idx_mag, idx_grad
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject


def rescale_meg_transform_outlier(arr):
    """
    same as rescale_meg, but also removes all values that are above [-1, 1]
    and rescales them to smaller values
    """

    arr = rescale_meg(arr)

    arr[arr < -1] *= 1e-2
    arr[arr > 1] *= 1e-2
    return arr


def rescale_meg(arr):
    """
    this tries to statically re-scale the values from Tesla to Nano-Tesla,
    such that most sensor values are between -1 and 1

    If possible, individual scaling is applied to magnetometers and
    gradiometers as both sensor types have a different sensitivity and scaling.

    Basically a histogram normalization between the two sensor types

    gradiometers  = *1e10
    magnetometers = *2e11
    """
    assert len(set(arr.shape) & set([306, 204, 102]))>0, f'Probably not the right amount of channels? {arr.shape=}'
    # some sanity check, if these
    if arr.min() < -1e-6 or arr.max() > 1e-6:
        warnings.warn(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
        raise Exception(
            "arr min/max are not in MEG scale, no rescaling applied: {arr.min()} / {arr.max()}"
        )
    arr = np.array(arr)
    grad_scale = 1e10
    mag_scale = 2e11

    # reshape to 3d to make indexing uniform for all types
    # will be put in its original shape later
    orig_shape = arr.shape
    arr = np.atleast_3d(arr)

    # heuristic to find which dimension is likely the sensor dimension
    for meg_type in [306, 204, 102]:  # mag+grad or grad or mag
        dims = [d for d, size in enumerate(arr.shape) if size % meg_type == 0]
        # how many copies do we have of the sensors?
        stacks = [
            size // meg_type for d, size in enumerate(arr.shape) if size % meg_type == 0
        ]
        if len(dims) > 0:
            break

    if len(dims) != 1:
        warnings.warn(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        raise Exception(
            f"Several or no matching dimensions found for sensor dimension: {arr.shape}"
            " will simply reshape everything with grad_scale."
        )
        return arr.reshape(*orig_shape) * grad_scale
    sensor_dim = dims[0]
    n_stack = stacks[0]

    if meg_type == 306:
        slicer_grad = [slice(None) for _ in range(3)]
        slicer_grad[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_grad for i in range(n_stack)]
        )
        arr[tuple(slicer_grad)] *= grad_scale
        slicer_mag = [slice(None) for _ in range(3)]
        slicer_mag[sensor_dim] = np.hstack(
            [(i * meg_type) + idx_mag for i in range(n_stack)]
        )
        arr[tuple(slicer_mag)] *= mag_scale

    if meg_type == 204:
        arr *= grad_scale

    if meg_type == 102:
        arr *= mag_scale

    return arr.reshape(*orig_shape)


def robust_scale_nd(arr, axis=None, with_centering=True, with_scaling=True,
                    quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
    """Robust-scale an nD array along a specified axis using sklearn's robust_scale.

    Args:
        arr: Array-like of any shape.
        axis: Axis along which to compute median/IQR. If None, scale over all elements.

    Returns:
        numpy.ndarray with the same shape as arr, robust-scaled along the given axis.
    """

    x = np.asarray(arr)

    # Global scaling over all elements
    if axis is None:
        X = x.reshape(-1, 1)
        Y = _robust_scale(X, axis=0, with_centering=with_centering,
                         with_scaling=with_scaling, quantile_range=quantile_range,
                         copy=copy, unit_variance=unit_variance)
        return Y.reshape(x.shape)

    # Normalize axis to be positive
    axis = axis % x.ndim

    # 1-D: scale over the only axis
    if x.ndim == 1:
        return _robust_scale(x.reshape(-1, 1), axis=0,
                            with_centering=with_centering, with_scaling=with_scaling,
                            quantile_range=quantile_range, copy=copy,
                            unit_variance=unit_variance).reshape(x.shape)

    # 2-D: delegate to sklearn with the chosen axis
    if x.ndim == 2:
        return _robust_scale(x, axis=axis, with_centering=with_centering,
                            with_scaling=with_scaling, quantile_range=quantile_range,
                            copy=copy, unit_variance=unit_variance)

    # nD (n >= 3): flatten all non-target axes into columns, scale along the target axis
    x_moved = np.moveaxis(x, axis, 0)
    L = x_moved.shape[0]
    X2d = x_moved.reshape(L, -1)

    Y2d = _robust_scale(X2d, axis=0, with_centering=with_centering,
                       with_scaling=with_scaling, quantile_range=quantile_range,
                       copy=copy, unit_variance=unit_variance)

    y_moved = Y2d.reshape(x_moved.shape)
    return np.moveaxis(y_moved, 0, axis)


def sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"]):
    """
    Checks that the first channel of `channels` is actually containing the
    most ECG events. Comparison is done by  mne.preprocessing.find_ecg_events,
    the channel with the lowest standard deviation between the intervals
    of heartbeats (the most regularly found QRSs) should be the ECG channel

    Parameters
    ----------
    raw : mne.Raw
        a MNE raw object.
    channels : list
        list of channel names that should be compared. First channel
        is the channel that should contain the most ECG events.
        The default is ['BIO001', 'BIO002', 'BIO003'].

    Returns
    -------
    bool
        True if ECG. Assertionerror if not.

    """
    stds = {}
    for ch in channels:
        x = mne.preprocessing.find_ecg_events(raw, ch_name=ch, verbose=False)
        t = x[0][:, 0]
        stds[ch] = np.std(np.diff(t))
    assert (
        np.argmin(stds.values()) == 0
    ), f"ERROR: {channels[0]} should be ECG, but did not have lowest STD: {stds}"
    return True


def load_events(file, event_ids=None):
    """retrieve event markers in chronological order from a mne readable file
    Parameters
    ----------
    file : str
        filepath of raw file, e.g. .fif or .edf.
    event_ids : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    np.ndarray
        mne.events array (n,3) -> (time, duration, event_id).

    """

    raw = mne.io.read_raw(file)
    min_duration = 3/raw.info['sfreq'] # our triggers are ~5ms long
    events = mne.find_events(raw, min_duration=min_duration,
                                 consecutive=False, verbose='WARNING')
    if event_ids is None:
        event_ids = np.unique(events[:,2])
    event_mask = [e in event_ids for e in events[:,2]]

    return events[event_mask,:]





def repair_epochs_autoreject(raw, epochs, ar_file, picks="meg"):
    """runs autorejec with default parameters on chosen picks

    Parameters
    ----------
    raw : mne.Raw
        the raw object of which the epochs where extracted from.
    epochs : mne.Epochs
        epochs object, chunked mne raw file.
    ar_file : str
        file location where the autoreject results should be saved.
    picks : str/list, optional
        which channels to run autoreject on. The default is "meg".

    Returns
    -------
    epochs_repaired : mne.Epochs
        the epochs with all repaired and remove epochs.

    """
    raise Exception('this function needs some cleanup')
    from utils import get_id

    # if precomputed solution exists, load it instead
    epochs_repaired_file = f"{ar_file[:-11]}.epochs"
    if os.path.exists(epochs_repaired_file):
        logging.info(f"Loading repaired epochs from {epochs_repaired_file}")
        epochs_repaired = mne.read_epochs(epochs_repaired_file, verbose="ERROR")
        return epochs_repaired

    # apply autoreject on this data to automatically repair
    # artefacted data points

    if os.path.exists(ar_file):
        logging.info(f"Loading autoreject pkl from {ar_file}")
        clf = read_auto_reject(ar_file)
    else:
        from utils import json_dump

        logging.info(f"Calculating autoreject pkl solution and saving to {ar_file}")
        json_dump({"events": epochs.events[:, 2].astype(np.int64)}, ar_file + ".json")
        clf = AutoReject(
            picks=picks, n_jobs=-1, verbose=False, random_state=get_id(ar_file)
        )
        clf.fit(epochs)
        clf.save(ar_file, overwrite=True)

    logging.info("repairing epochs")
    epochs_repaired, reject_log = clf.transform(epochs, return_log=True)

    ar_plot_dir = f"{settings.plot_dir}/autoreject/"
    os.makedirs(ar_plot_dir, exist_ok=True)

    event_ids = epochs.events[:, 2].astype(np.int64)
    arr_hash = hash_array(event_ids)

    n_bad = np.sum(reject_log.bad_epochs)
    arlog = {
        "mode": "repair & reject",
        "ar_file": ar_file,
        "bad_epochs": reject_log.bad_epochs,
        "n_bad": n_bad,
        "perc_bad": n_bad / len(epochs),
        "event_ids": event_ids,
    }

    subj = f"DSMR{get_id(ar_file)}"
    plt.maximize = False
    fig = plt.figure(figsize=[10, 10])
    ax = fig.subplots(1, 1)
    fig = reject_log.plot("horizontal", ax=ax, show=False)
    ax.set_title(f"{subj=} {n_bad=} event_ids={set(event_ids)}")
    fig.savefig(
        f"{ar_plot_dir}/{subj}_{os.path.basename(raw.filenames[0])}-{arr_hash}.png"
    )
    plt.close(fig)

    log_append(
        raw.filenames[0],
        f"autoreject_epochs event_ids={set(event_ids)} n_events={len(event_ids)}",
        arlog,
    )
    print(f"{n_bad}/{len(epochs)} bad epochs detected")
    epochs_repaired.save(epochs_repaired_file, verbose="ERROR")
    logging.info(f"saved repaired epochs to {epochs_repaired_file}")

    return reject_log, epochs_repaired
