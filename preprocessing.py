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
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject

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
