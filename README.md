# meg_utils

Python functions for everything MEG related. Simply to prevent having a copy of those files in all my repositories

The idea is to have this repo as a submodule in your project folder and import functions from there. This way, projects can keep a specific version of the repository as submodules, while the repo is continuously improved.

## Function Overview

### `conversion.py`

- `fif2edf(fif_file, chs=None, edf_file=None)`: Convert a FIF file to an EDF file using pyedflib.

### `data.py`

- `load_events(file, event_ids=None)`: retrieve event markers in chronological order from a mne readable file
- `load_meg(file, sfreq=100, ica=None, filter_func="lambda x:x", verbose="ERROR")`: Load MEG data and applies preprocessing to it (resampling, filtering, ICA)
- `load_epochs(file, sfreq=100, event_ids=None, event_filter=None, tmin=-0.1, tmax=0.5, ica=None, autoreject=True, filter_func="lambda x:x", picks="meg")`: Load data from FIF file and return into epochs given by MEG triggers.
- `load_segments(file, sfreq=100, markers=[[10, 11]], picks='meg', filter_func='lambda x:x', ica=None, verbose='ERROR')`: Load interval of data between two markers

### `decoding.py`

- `save_clf(clf, filename, save_json=True, metadata=None)`: Saves a scikit-learn classifier to a compressed pickle file, with an optional JSON sidecar containing parameters and training code metadata.
- `cross_validation_across_time(data_x, data_y, clf, add_null_data=False, n_jobs=-2, plot_confmat=False, title_add="", ex_per_fold=2, simulate=False, subj="", tmin=-0.1, tmax=0.5, sfreq=100, return_probas=False, verbose=True)`: Perform cross-validation across time on the given dataset.
- `decoding_heatmap_transfer(clf, data_x, data_y, test_x, test_y, range_t=None)`: create a heatmap of decoding by varying training and testing time of two independent samples
- `decoding_heatmap_generalization(clf, data_x, data_y, ex_per_fold=4, n_jobs=8, range_t=None)`: using cross validation, create a heatmap of decoding by varying training and testing times
- `train_predict(train_x, train_y, test_x, clf, neg_x=None, proba=False)`: Train a classifier with the given data and return predictions on test data.
- `get_channel_importances(data_x, data_y, n_folds=5, n_trees=500, **kwargs)`: returns the importance of features from a RandomForest cross-validation
- `to_long_df(arr, columns=None, value_name='value', **col_labels)`: Convert an N-D numpy array to a long-format DataFrame; include only labeled dims.
- `stratify(X, y, strategy='undersample', random_state=None, verbose=False)`: Balance a dataset by over- or undersampling.
- `TimeEnsembleVoting(base_estimator, voting='hard', weights=None, n_jobs=None, flatten_transform=True)`: TimeVotingClassifier trains one clone of a base estimator on each time slice of a 3D input (n_samples, n_features, n_times).
- `LogisticRegressionOvaNegX(base_clf=None, penalty="l1", C=1.0, solver="liblinear", max_iter=1000, neg_x_ratio=1.0)`: one vs all logistic regression classifier including negative examples.

### `misc.py`

- `list_files(path, exts=None, patterns=None, relative=False, recursive=False, subfolders=None, only_folders=False, max_results=None, case_sensitive=False)`: will make a list of all files with extention exts (list) found in the path and possibly all subfolders and return a list of all files matching this pattern
- `choose_file(default_dir=None, default_file=None, exts='txt', title='Choose file', mode='open', multiple=False)`: Open a file chooser dialog using tkinter.
- `string_to_seed(string)`: get a reproducible seed from any string, to set a random seed
- `hash_array(arr, length=8, dtype=np.int64)`: create a hash for any array by doing a full hash of the hexdigest
- `hash_md5(input_string, length=8)`: make a persistent md5 hash from a string
- `get_ch_neighbours(ch_name, n=9, return_idx=False, layout_name='Vectorview-all', plot=False)`: retrieve the n neighbours of a given MEG channel location.
- `low_priority()`: Set the priority of the process to below-normal (cross platform).

### `pipeline.py`

- `DataPipeline(steps, memory=None, verbose=False)`: A pipeline of data processing steps for MNE objects, extending scikit-learn's Pipeline class.
- `CustomStep(func, **kwargs)`: A custom processing step that applies a user-defined function.
- `LoadRawStep(preload=True, verbose=None, **kwargs)`: Load a raw data file from disk.
- `FilterStep(l_freq, h_freq, picks=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=None, method='fir', iir_params=None, phase='zero', fir_window='hamming', fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=None)`: Apply a band-pass filter to the data using MNE's filtering functions.
- `ResampleStep(sfreq, npad='auto', window='auto', stim_picks=None, n_jobs=None, events=None, pad='auto', method='fft', verbose=None)`: Resample the data to a new sampling frequency.
- `CropStep(id_start, id_stop, events=None, min_length_sanity=0, verbose=None)`: Crop a raw MNE object between two event markers.
- `EpochingStep(events=None, event_id=None, tmin=-0.2, tmax=0.5, baseline=(None, 0), picks=None, preload=True, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None, detrend=None, on_missing='error', reject_by_annotation=False, metadata=None, verbose=None)`: Create epochs from the raw data.
- `NormalizationStep(norm_func, axis=None, picks=None)`: Apply a normalization function to the data.
- `ToArrayStep(X=True, y=True, picks=None, verbose=None)`: Convert MNE objects to NumPy arrays.
- `ICAStep(n_components=None, method='fastica', random_state=97, max_iter='auto', decim=3, reject_by_annotation=True, tstep=0.02, verbose=None)`: No docstring available.
- `StratifyStep(strategy='undersample', random_state=None, verbose=None)`: No docstring available.

### `plotting.py`

- `plot_sensors(values, layout='auto', positions=None, title="Sensors active", mode="size", color=None, ax=None, vmin=None, vmax=None, cmap="Reds", **kwargs)`: Plot sensor positions with markers representing various data values.
- `make_sensor_importance_gif(output_filename, data_x=None, data_y=None, importances=None, accuracies = None, layout='auto', tmin=None, tmax=None, n_jobs=-1, n_folds=10, fps=0.2)`: No docstring available.
- `circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, **kwargs)`: Produce a circular histogram of angles on ax.
- `make_fig(n_axs=30, bottom_plots=2, no_ticks=False, suptitle="", xlabel="Timepoint", ylabel="", figsize=None, despine=True)`: helper function to create a grid space with RxC rows and a large row with two axis on the bottom
- `savefig(fig, file, tight=True, despine=True, **kwargs)`: Save a Matplotlib figure to a specified file with optional adjustments.
- `normalize_lims(axs, which='xy')`: Synchronize axis and/or color (clim) limits across a collection of Matplotlib Axes.
- `highlight_cells(mask, ax, color='r', linewidth=1, linestyle='solid')`: Draws borders around the true entries of the mask array on a heatmap plot.

### `preprocessing.py`

- `rescale_meg_transform_outlier(arr)`: same as rescale_meg, but also removes all values that are above [-1, 1] and rescales them to smaller values
- `rescale_meg(arr)`: this tries to statically re-scale the values from Tesla to Nano-Tesla, such that most sensor values are between -1 and 1
- `robust_scale_nd(arr, axis=None, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)`: Robust-scale an nD array along a specified axis using sklearn's robust_scale.
- `sanity_check_ECG(raw, channels=["BIO001", "BIO002", "BIO003"])`: Checks that the first channel of `channels` is actually containing the most ECG events.
- `load_events(file, event_ids=None)`: retrieve event markers in chronological order from a mne readable file
- `repair_epochs_autoreject(raw, epochs, ar_file, picks="meg")`: runs autorejec with default parameters on chosen picks

### `sigproc.py`

- `resample(array, o_sfreq, t_sfreq, n_jobs=-1, verbose=False)`: Resample EEG data using MNE.
- `bandpass(data, lfreq=None, ufreq=None, sfreq=100, verbose=False, **kwargs)`: Apply bandpass filter using MNE.
- `notch(data, freqs=None, notch_widths=None, sfreq=100, verbose=False, n_jobs=-1, **kwargs)`: Apply notch filter using MNE RawArray.
- `get_ch_neighbours(ch_name, n=9, return_idx=False, plot=False)`: retrieve the n neighbours of a given electrode location.
- `estimate_peak_alpha_freq(raw_or_fname, picks=None, alpha_band=(7.0, 13.0), spec_range=(2.0, 30.0), bandwidth=2.0, notch=(50,), hp=1.0, lp=40.0, use_specparam=True, specparam_kwargs=None, method="peak", plot=False, verbose=True)`: Estimate peak-alpha frequency (PAF) from an MEG recording.
- `get_alpha_peak(raw_or_data, sfreq: float | None = None, alpha_bounds: tuple[float, float] = (7, 14), return_spectrum: bool = False, plot_spectrum: bool = False)`: Alpha-peak finder for MNE Raw, (data, sfreq) tuples, or plain ndarrays.
- `get_alpha_phase(raw, alpha_peak, bandwidth=1.5)`: No docstring available.
- `create_oscillation(hz, sfreq=100, n_samples=None, n_seconds=None, phi_rad=None, phi_deg=None, amp=1.0)`: Generate a sinusoidal waveform with amplitude modulation per cycle.
- `wave_speed_cm(phases: np.ndarray, idx_source: int, pos: np.ndarray, freq_hz = None)`: Estimate the phase-velocity (cm s⁻¹) of a travelling cortical wave from MEG phase angles.
- `fit_curve(data, data_sfreq=1 / 1.25, *,  model=curves.gaussian, curve_sfreq=100, curve_params, plot_fit=False)`: Fit a parametric curve to 1-D data with L-BFGS-B.
- `interpolate(times, data, n_samples=None, kind='linear', axis=-1)`: interpolate data sampled at times to evenly spaced
- `polyfit(times, data, n_samples=None, degree=3, axis=-1)`: Fit a polynomial to data sampled at times and evaluate it at evenly spaced points.
- `find_phase_reversals(phases, threshold=np.pi/2)`: Find phase reversals in a 1D array of phase values.
