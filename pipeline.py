# -*- coding: utf-8 -*-
"""
Module implementing a data processing pipeline for MNE (MEG/EEG) data using scikit-learn conventions.

Created on Wed Nov 20 10:31:14 2024

@author: Simon
"""

import pathlib
import logging
import inspect
import mne
import warnings
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


class DataPipeline(Pipeline):
    """
    A pipeline of data processing steps for MNE objects, extending scikit-learn's Pipeline class.

    Parameters
    ----------
    steps : list of tuple
        List of (name, transform) tuples that are chained, in order.
    memory : str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default, no caching is performed.
    verbose : bool, default=False
        If True, logs the time elapsed while fitting each step.

    Attributes
    ----------
    named_steps : sklearn.utils.Bunch
        Dictionary-like object, with attribute access for retrieving any step parameter by name.

    Notes
    -----
    This pipeline checks that the output type of each step matches the input type of the next step.
    """

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.check_steps(verbose=False)

    def check_steps(self, verbose=True):
        """
        Check that the output type of each step matches the input type of the next step.

        Parameters
        ----------
        verbose : bool, default=True
            If True, prints the steps and types being checked.

        Raises
        ------
        AssertionError
            If the output type of a step does not match the input type of the next step.
        """
        if verbose:
            desc_in, func_in = self.steps[0]
            print(
                0, f'{func_in.type_in[0]}\t->\t({func_in.__class__.__name__})')

        for i, (desc_out, func_out) in enumerate(self.steps[:-1]):
            desc_in, func_in = self.steps[i + 1]
            match = False
            type_out = f if isinstance(
                f := func_out.type_out[0], tuple) else (f,)
            type_in = f if isinstance(f := func_in.type_in[0], tuple) else (f,)

            for otype in type_out:
                for itype in type_in:
                    if issubclass(otype, itype) or issubclass(itype, otype):
                        match = True
            assert match, (
                f'({func_out.__class__.__name__})\t->\t{func_out.type_out[0]} '
                f'!= {func_in.type_in[0]}\t->\t({func_in.__class__.__name__})'
            )
            if not verbose:
                continue
            print(
                i,
                f'({func_out.__class__.__name__})\t->\t{func_out.type_out[0]}'
                f'\t->\t({func_in.__class__.__name__})',
            )

        if verbose:
            (desc_out, func_out) = self.steps[-1]
            print(
                i + 1, f'({func_out.__class__.__name__})\t->\t{func_out.type_out[0]}')

    def set_params_all(self, overwrite_param=False, **params):
        """
        Set parameters for all steps that implement these parameters.

        Parameters
        ----------
        overwrite_param : bool, default=False
            If True, parameters will be overwritten even if they have been explicitly set.
            If False, only parameters at their default values will be updated.
        **params : dict
            Parameter names and values to set.

        Raises
        ------
        ValueError
            If a parameter is not found in any of the pipeline steps.
        """
        for key, val in params.items():
            found = False
            # Iterate over each step in the pipeline
            for desc, func in self.steps:
                if key in func.get_params():
                    # Get default parameter values from the __init__ signature
                    signature = inspect.signature(func.__init__)
                    default_params = {
                        k: p.default
                        for k, p in signature.parameters.items()
                        if p.default is not inspect.Parameter.empty
                    }
                    current_params = func.get_params()
                    # Check if the parameter has been explicitly changed
                    if key in default_params:
                        param_changed = current_params[key] != default_params[key]
                    else:
                        # If default value is not available, assume parameter is unchanged
                        param_changed = False
                    # Set parameter if not changed or if overwrite is True
                    if overwrite_param or not param_changed:
                        func.set_params(**{key: val})
                        if self.verbose:
                            logging.info(
                                f'Setting parameter {key}={val} for {desc}={func}')
                    found = True
            # Check if the parameter is in the pipeline itself
            if key in self.get_params():
                signature = inspect.signature(self.__class__.__init__)
                default_params = {
                    k: p.default
                    for k, p in signature.parameters.items()
                    if p.default is not inspect.Parameter.empty
                }
                current_params = self.get_params()
                if key in default_params:
                    param_changed = current_params[key] != default_params[key]
                else:
                    param_changed = False
                if overwrite_param or not param_changed:
                    self.set_params(**{key: val})
                found = True
            if not found:
                raise ValueError(
                    f'Did not find parameter "{key}" in any of the pipeline steps: {self.steps}'
                )


class BaseStep(TransformerMixin, BaseEstimator):
    """
    Base class for all processing steps in the pipeline.

    Provides a fit and transform method to be compatible with scikit-learn pipelines.
    Subclasses should implement the _transform method.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description.
    type_out : tuple
        Expected output types and a description.

    Notes
    -----
    Ensures type checking before and after transformation.
    """

    def fit(self, X, y=None):
        """
        Fit method required by scikit-learn's TransformerMixin.

        Parameters
        ----------
        X : object
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the input data and check types.

        Parameters
        ----------
        X : object
            The input data.

        Returns
        -------
        result : object
            The transformed data.

        Raises
        ------
        AssertionError
            If the input or output types are not as expected.
        """
        type_in, type_in_desc = self.type_in
        type_out, type_out_desc = self.type_out

        assert isinstance(X, type_in), (
            f'Input of {self} is expected to be {type_in} ({type_in_desc}) '
            f'but was {type(X)}'
        )
        result = self._transform(X)
        assert isinstance(result, type_out), (
            f'Output of {self} is expected to be {type_out} ({type_out_desc}) '
            f'but was {type(result)}'
        )
        return result


class CustomStep(BaseStep):
    """
    A custom processing step that applies a user-defined function.

    Parameters
    ----------
    func : callable
        The function to apply to the data.
    **kwargs : dict
        Additional keyword arguments to pass to the function.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description.
    type_out : tuple
        Expected output types and a description.
    """

    def __init__(self, func, **kwargs):
        self.type_in = (object,), 'anything'
        self.type_out = (object,), 'anything'
        self.func = func
        self.kwargs = kwargs

    def _transform(self, X):
        """
        Apply the custom function to the input data.

        Parameters
        ----------
        X : object
            The input data.

        Returns
        -------
        result : object
            The result of applying the custom function.
        """
        res = self.func(X, **self.kwargs)
        return res


class LoadRawStep(BaseStep):
    """
    Load a raw data file from disk.

    Parameters
    ----------
    preload : bool or str, default=True
        Whether to preload the data into memory.
    verbose : bool or str or None, optional
        Verbosity level.
    **kwargs : dict
        Additional keyword arguments to pass to the MNE `read_raw` function.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Location of the file').
    type_out : tuple
        Expected output types and a description ('Loaded raw file with nothing done').

    Notes
    -----
    This step loads raw data using MNE's `read_raw` function.
    """

    def __init__(self, *, preload=True, verbose=None, **kwargs):
        self.type_in = (str, pathlib.Path), 'Location of the file'
        self.type_out = (mne.io.BaseRaw,), 'Loaded raw file with nothing done'

        # Set parameters
        self.kwargs = kwargs

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, X):
        """
        Load the raw data file.

        Parameters
        ----------
        X : str or pathlib.Path
            The filename or path to the raw data file.

        Returns
        -------
        raw : mne.io.BaseRaw
            The loaded raw data object.
        """
        # Combine parameters from self and kwargs
        params = self.get_params()
        params.update(self.kwargs)
        # Determine the appropriate read function based on file extension
        fname = pathlib.Path(X)
        raw = mne.io.read_raw(fname, **params)

        return raw


class FilterStep(BaseStep):
    """
    Apply a band-pass filter to the data using MNE's filtering functions.

    Parameters
    ----------
    l_freq : float | None
        Low frequency of the filter (in Hz). Can be None for low-pass filter.
    h_freq : float | None
        High frequency of the filter (in Hz). Can be None for high-pass filter.
    picks : str | list | None, optional
        Channels to include. None means all channels.
    filter_length : str | int | None, optional
        Length of the filter.
    l_trans_bandwidth : float | str, optional
        Width of the transition band at the low cut-off frequency in Hz.
    h_trans_bandwidth : float | str, optional
        Width of the transition band at the high cut-off frequency in Hz.
    n_jobs : int | None, optional
        Number of jobs to run in parallel.
    method : str, optional
        Filtering method to use ('fir', 'iir').
    iir_params : dict | None, optional
        Parameters for IIR filtering.
    phase : str, optional
        Phase of the filter ('zero', 'zero-double', 'minimum').
    fir_window : str, optional
        Window to use in FIR design.
    fir_design : str, optional
        FIR design method.
    skip_by_annotation : list | str, optional
        Annotations to skip during filtering.
    pad : str, optional
        Type of padding to use.
    verbose : bool | str | None, optional
        Verbosity level.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Any MNE object that implements .filter').
    type_out : tuple
        Expected output types and a description.

    Notes
    -----
    This step applies filtering using MNE's built-in methods.
    """

    def __init__(
        self,
        l_freq,
        h_freq,
        picks=None,
        filter_length='auto',
        l_trans_bandwidth='auto',
        h_trans_bandwidth='auto',
        n_jobs=None,
        method='fir',
        iir_params=None,
        phase='zero',
        fir_window='hamming',
        fir_design='firwin',
        skip_by_annotation=('edge', 'bad_acq_skip'),
        pad='reflect_limited',
        verbose=None,
    ):
        self.type_in = (mne.filter.FilterMixin,
                        ), 'Any MNE object that implements .filter'
        self.type_out = (mne.io.BaseRaw,), ''

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, obj):
        """
        Apply the filter to the input data.

        Parameters
        ----------
        obj : mne.io.BaseRaw | mne.Epochs
            The data object to filter.

        Returns
        -------
        filtered_obj : mne.io.BaseRaw | mne.Epochs
            The filtered data object.
        """
        return obj.filter(**self.get_params())


class ResampleStep(BaseStep):
    """
    Resample the data to a new sampling frequency.

    Parameters
    ----------
    sfreq : float
        New sampling rate to resample the data to.
    npad : str | int, optional
        Amount to pad at the start and end of the data.
    window : str | tuple, optional
        Window to use in resampling.
    stim_picks : array_like of int | None, optional
        Stimulus channels to exclude from resampling.
    n_jobs : int | None, optional
        Number of jobs to run in parallel.
    events : array | None, optional
        Events to update during resampling.
    pad : str | None, optional
        Type of padding to use.
    method : str, optional
        Resampling method to use ('fir', 'fft').
    verbose : bool | str | None, optional
        Verbosity level.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Any MNE Raw object that can be resampled').
    type_out : tuple
        Expected output types and a description ('Resampled Raw object').

    Notes
    -----
    This step resamples the data to a lower sampling frequency.
    """

    def __init__(
        self,
        sfreq,
        *,
        npad='auto',
        window='auto',
        stim_picks=None,
        n_jobs=None,
        events=None,
        pad='auto',
        method='fft',
        verbose=None,
    ):
        self.type_in = (
            mne.io.BaseRaw,), 'Any MNE Raw object that can be resampled'
        self.type_out = (mne.io.BaseRaw,), 'Resampled Raw object'

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, raw):
        """
        Resample the raw data to a lower sampling frequency.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data object to resample.

        Returns
        -------
        raw_resampled : mne.io.BaseRaw
            The resampled raw data object.

        Raises
        ------
        ValueError
            If the new sampling frequency is higher than the original.
        """
        # Check that sfreq is lower than raw.info['sfreq']
        raw_sfreq_rounded = np.round(raw.info['sfreq'], 2)
        if self.sfreq > raw_sfreq_rounded:
            raise ValueError(
                f"Resampling frequency ({self.sfreq} Hz) must be lower than original sampling "
                f"frequency ({raw.info['sfreq']} Hz)."
            )
        if raw_sfreq_rounded == self.sfreq:
            logging.info(
                'Sample frequency is already what is requested, skipping ResampleStep')
            return raw
        # Resample the raw data
        raw_resampled = raw.copy().resample(**self.get_params())
        return raw_resampled


class EpochingStep(BaseStep):
    """
    Create epochs from the raw data.

    Parameters
    ----------
    event_id : dict | list | int | None
        The id(s) of the event(s) to consider.
    tmin : float
        Start time before event (in seconds).
    tmax : float
        End time after event (in seconds).
    baseline : None or tuple of length 2
        The time interval to consider as baseline. If None, no baseline correction is applied.
    picks : str | list | None, optional
        Channels to include. None means all channels.
    preload : bool, default=True
        Whether to preload the data.
    reject : dict | None, optional
        Rejection parameters based on peak-to-peak amplitude.
    flat : dict | None, optional
        Rejection parameters based on flatness of the signal.
    proj : bool, optional
        Whether to apply projections.
    decim : int, optional
        Factor by which to decimate the data.
    reject_tmin : float | None, optional
        Start time to consider for rejection.
    reject_tmax : float | None, optional
        End time to consider for rejection.
    detrend : int | None, optional
        Whether to apply detrending.
    on_missing : str, optional
        What to do if an event is missing ('error', 'warn', 'ignore').
    reject_by_annotation : bool, optional
        Whether to reject according to annotations.
    metadata : pandas.DataFrame | None, optional
        The metadata DataFrame.
    verbose : bool | str | None, optional
        Verbosity level.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Any MNE Raw object to be epoched').
    type_out : tuple
        Expected output types and a description ('Epochs object created from raw').

    Notes
    -----
    This step creates epochs from the raw data using MNE's Epochs class.
    """

    def __init__(
        self,
        events=None,
        event_id=None,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        picks=None,
        preload=True,
        reject=None,
        flat=None,
        proj=True,
        decim=1,
        reject_tmin=None,
        reject_tmax=None,
        detrend=None,
        on_missing='error',
        reject_by_annotation=False,
        metadata=None,
        verbose=None,
    ):
        self.type_in = (mne.io.BaseRaw,), 'Any MNE Raw object to be epoched'
        self.type_out = (mne.Epochs,), 'Epochs object created from raw'

        assert (events is None) != (
            event_id is None), 'either events or event_id must be supplied'

        # Transformation to list, as MNE doesn't like it as an array
        if isinstance(event_id, np.ndarray) and event_id.ndim == 1:
            event_id = list(event_id)

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])


    def _transform(
        self,
        raw,
    ):
        """
        Create epochs from the raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data object.
        stim_channel : None | str | list of str, optional
            The name(s) of the stimulus channel(s).
        output : str, optional
            Type of output ('onset', 'offset', 'step').
        min_duration : float, optional
            Minimum duration of each event.
        shortest_event : int, optional
            Minimum number of samples for an event.
        mask : ndarray | None, optional
            Array for masking stimulus channels.
        uint_cast : bool, optional
            Whether to cast to unsigned integer.
        mask_type : str, optional
            Type of mask ('and', 'or').
        initial_event : bool, optional
            Whether to consider the initial event.
        verbose : bool | str | None, optional
            Verbosity level.

        Returns
        -------
        epochs : mne.Epochs
            The epochs created from the raw data.

        Notes
        -----
        Uses MNE's `find_events` function to find events and create epochs.
        """
        # Find events in the raw data
        if self.events is None:
            events = mne.find_events(raw)
        else:
            events = self.events
            # ad first_samp to events data. This might be buggy if you already
            # removed that beforehand
            events[:, 0] += raw.first_samp

        # Create epochs from the raw data
        params = self.get_params()
        params['events'] = events
        epochs = mne.Epochs(raw, **params)

        if self.verbose:
            logging.info(
                f'Epoching into {len(epochs)} epochs, event_ids={np.unique(events[:, 2])}'
            )
        return epochs


class NormalizationStep(BaseStep):
    """
    Apply a normalization function to the data.

    Parameters
    ----------
    norm_func : callable
        The normalization function to apply.
    axis : int | None
        Axis along which to normalize.
    picks : array_like | None
        Channels to include in normalization.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Any MNE obj or array').
    type_out : tuple
        Expected output types and a description ('Any MNE obj or array').

    Notes
    -----
    This step applies a custom normalization function to the data using public MNE APIs.
    """

    def __init__(self, norm_func, axis=None, picks=None):
        self.type_in = (np.ndarray, mne.io.BaseRaw,
                        mne.epochs.BaseEpochs), 'Any MNE obj or array'
        self.type_out = (np.ndarray, mne.io.BaseRaw,
                         mne.epochs.BaseEpochs), 'Any MNE obj or array'
        self.norm_func = norm_func
        self.axis = axis
        self.picks = picks

    def _transform(self, obj):
        """
        Apply the normalization function to the data.

        Parameters
        ----------
        obj : mne.io.BaseRaw | mne.Epochs | np.ndarray
            The data object to normalize.

        Returns
        -------
        obj_normalized : mne.io.BaseRaw | mne.Epochs | np.ndarray
            The normalized data object.

        Raises
        ------
        Exception
            If an unsupported object type is provided.
        """
        if isinstance(obj, (mne.io.BaseRaw, mne.epochs.BaseEpochs)):
            # Get the data using public API
            data = obj.get_data()
            # Apply normalization
            data_normalized = self.norm_func(data)
            # Reconstruct the MNE object with normalized data
            if isinstance(obj, mne.io.BaseRaw):
                obj_normalized = mne.io.RawArray(data_normalized, obj.info)
            else:  # mne.Epochs
                obj_normalized = mne.EpochsArray(
                    data_normalized, obj.info, events=obj.events,
                    event_id=obj.event_id, tmin=obj.tmin, verbose='ERROR'
                )
        elif isinstance(obj, np.ndarray):
            if self.picks is not None:
                raise ValueError(
                    f'picks={self.picks} has been supplied, but input is a NumPy array.'
                )
            obj_normalized = self.norm_func(obj)
        else:
            raise Exception(f'Unsupported object type: {type(obj)}')
        return obj_normalized


class ToArrayStep(BaseStep):
    """
    Convert MNE objects to NumPy arrays.

    Parameters
    ----------
    X : bool, default=True
        Whether to include the data array.
    y : bool, default=True
        Whether to include the event labels.
    picks : list | None
        Channels to pick.

    Attributes
    ----------
    type_in : tuple
        Expected input types and a description ('Any MNE object').
    type_out : tuple
        Expected output types and a description ('data_x, data_y' as a tuple).

    Notes
    -----
    This step extracts the data and labels from MNE objects.
    """

    def __init__(self, X=True, y=True, picks=None, verbose=None):
        self.type_in = (
            mne.io.BaseRaw, mne.epochs.BaseEpochs), 'Any MNE object'
        self.type_out = (tuple,), 'data_x, data_y'

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, obj):
        """
        Extract data and labels from the MNE object.

        Parameters
        ----------
        obj : mne.io.BaseRaw | mne.Epochs
            The data object from which to extract arrays.

        Returns
        -------
        data : tuple
            A tuple containing the data array and optionally the labels.

        Notes
        -----
        The data array is returned as the first element, and labels (if available) as the second element.
        """
        data = []
        if self.X:
            data_x = obj.get_data(picks=self.picks, verbose=self.verbose)
            data.append(data_x)
        if self.y:
            if hasattr(obj, 'events'):
                data_y = obj.events[:, 2]
                data.append(data_y)
            else:
                data.append(None)
        return tuple(data)


class ICAStep(BaseStep):
    def __init__(self, n_components=None, method='fastica', random_state=97, max_iter='auto', decim=3, reject_by_annotation=True, tstep=0.02, verbose=None):
        raise Exception('Step is from chatgpt - check first')
        self.type_in = mne.io.BaseRaw, 'Any MNE Raw object for ICA'
        self.type_out = mne.io.BaseRaw, 'Raw object after ICA applied'

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, raw):
        # Create ICA object with specified parameters
        ica = mne.preprocessing.ICA(n_components=self.n_components, method=self.method,
                                    random_state=self.random_state, max_iter=self.max_iter, verbose=self.verbose)
        # Select channels to include (e.g., MEG/EEG channels)
        picks = raw.pick(['eeg', 'meg'], exclude='bads')
        # Fit ICA
        ica.fit(raw, picks=picks, decim=self.decim,
                reject_by_annotation=self.reject_by_annotation, tstep=self.tstep)
        # Find EOG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        # Find ECG artifacts
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
        # Exclude the bad components
        ica.exclude = eog_indices + ecg_indices
        # Apply ICA correction
        raw_corrected = ica.apply(raw.copy())
        return raw_corrected


class StratifyStep(BaseStep):
    def __init__(self, strategy='undersample', random_state=None, verbose=None):
        warnings.warn('Step is from chatgpt - check first')
        self.type_in = (mne.epochs.BaseEpochs,
                        ), 'Epochs object to be stratified'
        self.type_out = (mne.epochs.BaseEpochs,), 'Stratified Epochs object'

        params = self._get_param_names()
        for name in params:
            setattr(self, name, locals()[name])

    def _transform(self, epochs):
        # Count the number of epochs per event type
        event_ids = epochs.event_id
        event_counts = {event: len(epochs[event]) for event in event_ids}

        rng = np.random.RandomState(self.random_state)

        if self.strategy == 'undersample':
            # Find the minimum count
            min_count = min(event_counts.values())
            if self.verbose:
                print(f"Undersampling to {min_count} epochs per event type.")
            # Select min_count epochs from each event type
            epochs_list = []
            for event in event_ids:
                these_epochs = epochs[event]
                indices = rng.choice(
                    len(these_epochs), size=min_count, replace=False)
                these_epochs = these_epochs[indices]
                epochs_list.append(these_epochs)
            # Concatenate the epochs
            epochs_balanced = mne.concatenate_epochs(epochs_list)
        elif self.strategy == 'oversample':
            # Find the maximum count
            max_count = max(event_counts.values())
            if self.verbose:
                print(f"Oversampling to {max_count} epochs per event type.")
            # Resample epochs to max_count for each event
            epochs_list = []
            for event in event_ids:
                these_epochs = epochs[event]
                n = len(these_epochs)
                if n < max_count:
                    # Randomly resample with replacement
                    indices = rng.choice(n, size=max_count, replace=True)
                    these_epochs = these_epochs[indices]
                epochs_list.append(these_epochs)
            # Concatenate the epochs
            epochs_balanced = mne.concatenate_epochs(epochs_list)
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Use 'undersample' or 'oversample'.")
        return epochs_balanced


# %% main / test

if __name__ == '__main__':
    filename = 'Z:/fastreplay-MEG-bids/sub-24/meg/sub-24_task-main_split-02_meg.fif'

    x = LoadRawStep()

    pipeline = DataPipeline(steps=[
        ('load raw', LoadRawStep()),
        ('filter', FilterStep(l_freq=1.0, h_freq=40.0)),
    ], verbose=True)
    pipeline.check_steps()
    pipeline.set_params_all(n_jobs=2, verbose='INFO')

    # pipeline.transform(filename)
