# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`meg_utils` is a shared Python utility library for MEG (Magnetoencephalography) analysis, designed to be used as a git submodule across multiple projects. It wraps MNE-Python and scikit-learn with MEG-specific conveniences.

## Commands

### Install (editable)
```bash
pip install -e .
pip install pytest  # for running tests
```

### Run all tests
```bash
pytest tests/ -v
```

### Run a single test file
```bash
pytest tests/test_decoding.py -v
```

### Run a single test class or method
```bash
pytest tests/test_misc.py::TestLongDfToArray::test_roundtrip_basic -v
```

### Sync all instances of meg_utils across Nextcloud repos
```bash
python _synchronize_meg_utils.py
```

## Repository Structure

The package lives under `meg_utils/` (the inner directory). The root also contains legacy top-level `.py` stubs (e.g., `decoding.py`, `plotting.py`) that are **not** the active source — always edit files under `meg_utils/`.

```
meg_utils/           ← Python package (the real source)
  __init__.py        ← imports all submodules + exposes Stop
  _constants.py      ← MEGIN TRIUX sensor index arrays (idx_grad, idx_mag)
  conversion.py      ← FIF↔EDF conversion
  data.py            ← load_meg, load_epochs, load_segments, load_events
  decoding.py        ← classifiers, CV, heatmaps, stratify, reduce_dimensions
  misc.py            ← file utilities, hashing, to_long_df / long_df_to_array
  pipeline.py        ← DataPipeline (sklearn-style MNE processing steps)
  plotting.py        ← sensor plots, GIF creation, figure helpers
  preprocessing.py   ← MEG rescaling, autoreject, robust scaling
  sigproc.py         ← filtering, resampling, alpha peak, oscillation tools
tests/
  test_decoding.py
  test_misc.py
  test_sigproc.py
_synchronize_meg_utils.py  ← pulls updates in all Nextcloud copies of this repo
```

## Architecture

### Module responsibilities

- **`data.py`**: High-level loaders that compose preprocessing into single calls (`load_meg`, `load_epochs`, `load_segments`). Accepts ICA, autoreject, custom filter functions, and event filtering.

- **`pipeline.py`**: sklearn-style `DataPipeline` extending `sklearn.pipeline.Pipeline`. Each step (`LoadRawStep`, `FilterStep`, `ResampleStep`, `CropStep`, `EpochingStep`, `NormalizationStep`, `ToArrayStep`) declares `type_in`/`type_out` tuples; `DataPipeline.check_steps()` validates the chain at construction time. Use `pipeline.set_params_all(n_jobs=4)` to propagate a parameter to all steps that accept it.
  > **Note**: `ICAStep` and `StratifyStep` raise warnings/exceptions — they are ChatGPT drafts that need manual verification before use.

- **`decoding.py`**: Time-resolved decoding utilities. Key patterns:
  - `cross_validation_across_time(data_x, data_y, clf)` expects `data_x` shape `(n_samples, n_features, n_timepoints)` and returns a long-format DataFrame of per-fold accuracy per timepoint.
  - `LogisticRegressionOvaNegX`: one-vs-all LR that accepts an external `neg_x` "null" class during `fit()`.
  - `TimeEnsembleVoting`: trains one classifier clone per time slice of a 3D array and aggregates via VotingClassifier.
  - `reduce_dimensions(data, axis, n_components)`: fits PCA along any axis and returns `(reduced_data, reducer)` where `reducer` can be called on new data.
  - `save_clf(clf, filename)`: saves `.pkl.gz` + JSON sidecar with hyperparams and calling code line.

- **`misc.py`**:
  - `to_long_df(arr, columns, value_name, **col_labels)` / `long_df_to_array(df, columns, value_name)`: round-trip conversion between N-D numpy arrays and long-format DataFrames. Fortran-order linearisation. Dimensions named `None`, `False`, or `'_'` are skipped.
  - `NumpyEncoder`: JSON encoder for numpy types (used in `save_clf`).
  - `Stop`: raises `KeyboardInterrupt` subclass that exits a script cleanly to REPL without traceback — use `raise Stop` instead of `raise SystemExit` in interactive sessions.
  - `telegram_callback`: decorator that sends Telegram messages on function begin/finish/error via `telegram_send`.

- **`sigproc.py`**: Array-level signal processing (no MNE objects required for most functions). Includes `bandpass`, `notch`, `resample`, `estimate_peak_alpha_freq`, `get_alpha_peak`, `fit_curve`, `interpolate`, `sliding_window`, `wave_speed_cm`.

- **`_constants.py`**: `idx_grad` and `idx_mag` — NumPy integer arrays of gradiometer and magnetometer channel indices for the MEGIN TRIUX 306-channel system (102 mags, 204 grads).

### Data conventions

- MEG epoch arrays follow the shape `(n_samples, n_channels, n_timepoints)` throughout.
- Default sampling frequency after preprocessing is **100 Hz**.
- Default epoch window: `tmin=-0.1`, `tmax=0.5` seconds.
- `cross_validation_across_time` requires **balanced classes** (equal samples per class) — use `stratify()` first if needed.

### Testing approach

Tests use `unittest.TestCase` (decoding, sigproc) and plain pytest classes (misc). They do not require real MEG data — all tests use synthetically generated numpy arrays. CI runs against Python 3.10, 3.12, and 3.14.
