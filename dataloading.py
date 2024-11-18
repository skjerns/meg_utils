# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:20:19 2024

@author: Simon Kern (@skjerns)
"""
import mne
from joblib import Memory, Parallel, delayed
import numpy as np

memory = Memory('xxx')

@memory.cache(verbose=0)
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


def load_meg(file, sfreq=100, ica=None, filter_func="lambda x:x", verbose="ERROR"):
    """
    Load MEG data and applies preprocessing to it (resampling, filtering, ICA)

    Parameters
    ----------
    file : str
        Which MEG file to load.
    sfreq : int, optional
        Resample to this sfreq. The default is 100.
    ica : int or bool, optional
        Apply ICA with the number of components as ICA. The default is None.

    filter_func : str, func, optional
        a lambda string or function that will be applied
        to filter the data. The default is 'lambda x:x' (no filtering).

    Returns
    -------
    raw, events : mne.io.Raw, np.ndarray
        loaded raw file and events for the file correctly resampled

    """

    @memory.cache(ignore=["verbose"])
    def _load_meg_ica(file, sfreq, ica, verbose):
        """loads MEG data, calculates artefacted epochs before"""
        from utils import get_id

        tstep = 2.0  # default ICA value for tstep
        raw = mne.io.read_raw_fif(file, preload=True, verbose=verbose)
        raw_orig = raw.copy()
        min_duration = 3 / raw.info["sfreq"]  # our triggers are ~5ms long
        events = mne.find_events(
            raw, min_duration=min_duration, consecutive=False, verbose=verbose
        )
        # resample if requested
        # before all operations and possible trigger jitter, exctract the events
        if sfreq and np.round(sfreq) != np.round(raw.info["sfreq"]):
            raw, events = raw.resample(sfreq, n_jobs=1, verbose=verbose, events=events)

        if ica:
            assert isinstance(ica, int), "ica must be of type INT"
            n_components = ica
            ica_fif = os.path.basename(file).replace(
                ".fif", f"-{sfreq}hz-n{n_components}.ica"
            )
            ica_fif = settings.cache_dir + "/" + ica_fif
            # if we previously applied an ICA with these components,
            # we simply load this previous solution
            if os.path.isfile(ica_fif):
                ica = read_ica(ica_fif, verbose="ERROR")
                assert (
                    ica.n_components == n_components
                ), f"n components is not the same, please delete {ica_fif}"
                assert (
                    ica.method == "picard"
                ), f"ica method is not the same, please delete {ica_fif}"
            # else we compute it
            else:
                ####### START OF AUTOREJECT PART
                # by default, apply autoreject to find bad parts of the data
                # before fitting the ICA
                # determine bad segments, so that we can exclude them from
                # the ICA step, as is recommended by the autoreject guidelines
                logging.info("calculating outlier threshold for ICA")
                equidistants = mne.make_fixed_length_events(raw, duration=tstep)
                # HP filter data as recommended by the autoreject codebook
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=verbose)
                epochs = mne.Epochs(
                    raw_hp,
                    equidistants,
                    tmin=0.0,
                    tmax=tstep,
                    baseline=None,
                    verbose="WARNING",
                )
                reject = get_rejection_threshold(
                    epochs, verbose=verbose, cv=10, random_state=get_id(file)
                )
                epochs.drop_bad(reject=reject, verbose=False)
                log_append(
                    file,
                    "autoreject_raw",
                    {
                        "percentage_removed": epochs.drop_log_stats(),
                        "bad_segments": [x != () for x in epochs.drop_log],
                    },
                )
                ####### END OF AUTOREJECT PART

                ####### START OF ICA PART
                # use picard that simulates FastICA, this is specified by
                # setting fit_params to ortho and extended=True
                ica = ICA(
                    n_components=n_components,
                    method="picard",
                    verbose="WARNING",
                    fit_params=dict(ortho=True, extended=True),
                    random_state=get_id(file),
                )
                # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
                # we later apply the ICA components to the not-filtered signal
                raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="WARNING")
                ica.fit(raw_hp, picks="meg", reject=reject, tstep=tstep)
                ica.save(ica_fif)  # save ICA to file for later loading
                ####### END OF ICA PART

            assert sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"])
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_orig, ch_name="BIO001", verbose="WARNING"
            )
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, threshold=2, ch_name=["BIO002", "BIO003"], verbose="WARNING"
            )
            emg_indices, emg_scores = ica.find_bads_muscle(raw_orig, verbose="WARNING")

            if len(ecg_indices) == 0:
                warnings.warn("### no ECG component found, is 0")
            if len(eog_indices) == 0:
                warnings.warn("### no EOG component found, is 0")
            components = list(set(ecg_indices + eog_indices + emg_indices))
            ica_log = {
                "ecg_indices": ecg_indices,
                "eog_indices": eog_indices,
                "emg_indices": emg_indices,
            }
            log_append(file, "ica", ica_log)

            ica.exclude = components
            raw = ica.apply(raw, verbose="WARNING")
        return raw, events

    raw, events = _load_meg_ica(file, sfreq=sfreq, ica=ica, verbose=verbose)

    # lamba functions don't work well with caching
    # so allow definition of lambda using strings
    # filtering is done after ICA.
    if filter_func != "lambda x:x":
        print("filtering")
    if isinstance(filter_func, str):
        filter_func = eval(filter_func)
    raw = filter_func(raw)
    return raw, events

@memory.cache(ignore=["n_jobs"])
def load_epochs_bands(
    file,
    bands,
    sfreq=100,
    event_ids=None,
    tmin=-0.1,
    tmax=0.5,
    ica=None,
    autoreject=True,
    picks="meg",
    event_filter=None,
    n_jobs=1,
):

    assert isinstance(bands, dict), f"bands must be dict, but is {type(bands)}"

    if len(bands) > 1 and autoreject:
        raise ValueError("If several bands are used, cannot reject epochs")
    log_append(
        file,
        "parameters_bands",
        {
            "file": file,
            "sfreq": sfreq,
            "ica": ica,
            "event_ids": event_ids,
            "autoreject": autoreject,
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "bands": bands,
            "event_filter": event_filter,
        },
    )

    if n_jobs < 0:
        n_jobs = len(bands) + 1 - n_jobs
    data = Parallel(n_jobs=n_jobs)(
        delayed(load_epochs)(
            file,
            sfreq=sfreq,
            filter_func=f"lambda x: x.filter({lfreq}, {hfreq}, verbose=False, n_jobs=-1)",
            event_ids=event_ids,
            tmin=tmin,
            tmax=tmax,
            ica=ica,
            event_filter=event_filter,
            picks=picks,
            autoreject=autoreject,
        )
        for lfreq, hfreq in bands.values()
    )
    data_x = np.hstack([d[0] for d in data])
    data_y = data[0][1]
    return (data_x, data_y)


def load_epochs(
    file,
    sfreq=100,
    event_ids=None,
    event_filter=None,
    tmin=-0.1,
    tmax=0.5,
    ica=None,
    autoreject=True,
    filter_func="lambda x:x",
    picks="meg",
):
    """
    Load data from FIF file and return into epochs given by MEG triggers.
    stratifies the classes, that means each class will have the same
    number of examples.
    """
    if event_ids is None:
        event_ids = list(range(1, 11))
    raw, events = load_meg(file, sfreq=sfreq, ica=ica, filter_func=filter_func)

    events_mask = [True if idx in event_ids else False for idx in events[:, 2]]
    events = events[events_mask, :]

    if event_filter:
        if isinstance(event_filter, str):
            event_filter = eval(event_filter)
        events = event_filter(events)

    data_x, data_y = make_meg_epochs(
        raw, events=events, tmin=tmin, tmax=tmax, autoreject=autoreject, picks=picks
    )

    # start label count at 0 not at 1, so first class is 0
    data_y -= 1

    return data_x, data_y


def load_segments(file, sfreq=100, markers=[[10, 11]], picks='meg',
                  filter_func='lambda x:x', ica=None, verbose='ERROR'):
    """
    Load interval of data between two markers

    Parameters
    ----------
    file : str
        FIF file to load.
    sfreq : int, optional
        frequency to which to downsample. The default is 100.
    markers : list of list of 2 ints, optional
        which trigger values to take the segment between. The default is [[10, 11]].
        the first and last occurence of the marker is taken as the segment length
    slicer : TYPE, optional
        DESCRIPTION. The default is None.
    picks : str or int, optional
        string or int for which channels to load. The default is 'meg'.
    filter_func : str, optional
        lambda string for filtering the segments. The default is 'lambda x:x'.
    ica : int, optional
        how many ICA components to discard. The default is None.
    verbose : str, optional
        MNE verbose marker. The default is 'ERROR'.

    Returns
    -------
    segments : TYPE
        DESCRIPTION.

    """
    if len(markers)>1:
        raise NotImplementedError('Check end of function, this doesnt work yet')

    # now get segments from data
    raw, events = load_meg(file, sfreq=sfreq, filter_func=filter_func, ica=ica, verbose=verbose)
    print(f'available events: {np.unique(events[:,2])}, looking for {markers}')

    data = raw.get_data(picks=picks)
    triggers = raw.get_data(picks='STI101')

    segments = []
    first_samp = raw.first_samp
    all_markers = np.unique(markers) if isinstance(markers, list) else []
    found_event_idx = [True if e in all_markers else False for e in events[:,2]]
    events = events[found_event_idx, :]

    if len(events)==0:
        warnings.warn(f'No matching events for {all_markers} found in {file}, taking 90% middle segment of file')
        tstart = int(len(raw)*0.05)
        tstop = int(len(raw)*0.95)
        markers = [[0, 1]]
        events = np.array([[tstart, 0, markers[0][0]], [tstop,0, markers[0][1]]])

    start_id, stop_id = markers[0]
    segtuples = np.array(list(zip(events[::2], events[1::2])))

    segments = []
    trigger_val = []
    for start, stop in segtuples:
        assert start[2]==start_id
        assert stop[2]==stop_id
        t_start = start[0] - first_samp
        t_end = stop[0] - first_samp
        seg = data[:, t_start:t_end]
        tpos = np.where(triggers[0, t_start:t_end])[0]
        tval = triggers[0, t_start:t_end][tpos]
        trigger_val.append(list(zip(tpos, tval)))
        segments.append(seg)
    lengths = [seg.shape[-1] for seg in segments]
    if not any(np.diff(np.unique(lengths))>2):
        segments = [seg[:,:min(lengths)] for seg in segments]
    return segments


@memory.cache
def load_segments_bands(file, bands, sfreq=100, markers=[[10, 11]], picks='meg',
                        ica=settings.default_ica_components, verbose='ERROR',
                        n_jobs=3):
    log_append(file, 'parameters_bands', {'file': file, 'sfreq': sfreq,
                                         'ica': ica, 'markers': markers,
                                         'picks': picks, 'bands': bands})

    data_x = Parallel(n_jobs=n_jobs)(delayed(load_segments)
            (file, sfreq=sfreq, markers=markers, ica=ica, picks=picks,
             filter_func=f'lambda x: x.filter({lfreq}, {hfreq}, verbose=False, n_jobs=-1)')
            for lfreq, hfreq in bands.values())
    data_x = np.array(data_x).squeeze()
    data_x = default_normalize(data_x)
    return data_x
