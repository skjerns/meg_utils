# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:26 2024

@author: Simon Kern (@skjerns)
"""
import sys
from pathlib import Path
from collections import namedtuple
from natsort import natsort_key
import warnings
import mne
import hashlib
import numpy as np
import json
import pandas as pd
import time
import inspect
import traceback
from functools import wraps
from html import escape as _esc

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Stop(KeyboardInterrupt):
    """gracefully exit a script and return to REPL without traceback

    usage:
        raise StopExecution
    """
    def _render_traceback_(self):
        print('Script execution stopped: ', self, end='')
        return []  # returning an empty list prevents the TypeError


def list_files(path, exts=None, patterns=None, relative=False, recursive=False,
               subfolders=None, only_folders=False, max_results=None,
               case_sensitive=False):
    """
    will make a list of all files with extention exts (list)
    found in the path and possibly all subfolders and return
    a list of all files matching this pattern

    :param path:  location to find the files
    :type  path:  str
    :param exts:  extension of the files (e.g. .jpg, .jpg or .png, png)
                  Will be turned into a pattern internally
    :type  exts:  list or str
    :param pattern: A pattern that is supported by pathlib.Path,
                  e.g. '*.txt', '**\rfc_*.clf'
    :type:        str
    :param fullpath:  give the filenames with path
    :type  fullpath:  bool
    :param subfolders
    :param return_strings: return strings, else returns Path objects
    :return:      list of file names
    :type:        list of str
    """
    def insensitive_glob(pattern):
        f = lambda c: '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
        return ''.join(map(f, pattern))

    if subfolders is not None:
        import warnings
        warnings.warn("`subfolders` is deprecated, use `recursive=` instead", DeprecationWarning)
        recursive = subfolders

    if isinstance(exts, str): exts = [exts]
    if isinstance(patterns, str): patterns = [patterns]

    p = Path(path)
    assert p.exists(), f'Path {path} does not exist'
    if patterns is None: patterns = []
    if exts is None: exts = []

    if not patterns and not exts:
        patterns = ['*']

    for ext in exts:
        ext = ext.replace('*', '')
        pattern = '*' + ext
        patterns.append(pattern.lower())

    # if recursiveness is asked, prepend the double asterix to each pattern
    if recursive: patterns = ['**/' + pattern for pattern in patterns]

    # collect files for each pattern
    files = []
    fcount = 0
    for pattern in patterns:
        if not case_sensitive:
            pattern = insensitive_glob(pattern)
        for filename in p.glob(pattern):
            if filename.is_file() and filename not in files:
                if only_folders:
                    continue
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results<=fcount:
                    break
            elif filename.is_dir() and only_folders and filename not in files:
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results<=fcount:
                    break


    # turn path into relative or absolute paths
    if relative:
        files = [file.relative_to(p) for file in files]

    # by default: return strings instead of Path objects
    files = [str(file) for file in files]
    files = set(files)  # filter duplicates
    return sorted(files, key=natsort_key)

def choose_file(default_dir=None, default_file=None, exts='txt',
                title='Choose file', mode='open', multiple=False):
    """
    Open a file chooser dialog using tkinter.

    Parameters
    ----------
    default_dir : str or None, optional
        Directory to open initially. If None, the current working directory is used.
    default_file : str or None, optional
        The default filename to use (only applicable in 'save' mode).
    exts : str or list of str, optional
        A file extension or a list of file extensions to filter the file types,
        e.g., 'txt' or ['txt', 'csv'].
    title : str, optional
        The title of the file dialog window.
    mode : {'open', 'save'}, optional
        The mode of the file dialog: 'open' to select existing files, 'save' to
        specify a file to save.
    multiple : bool, optional
        When True and mode is 'open', allows multiple file selection. Defaults
        to False.

    Returns
    -------
    str or list of str
        The selected file path(s). Returns a string if a single file is selected,
        or a list of strings if multiple files are selected.
        Returns None if no file is selected.

    Raises
    ------
    ValueError
        If an unknown mode is provided or if 'multiple' is True in 'save' mode.

    Notes
    -----
    This function creates a temporary Tkinter root window to display the file dialog.
    The root window is destroyed after the dialog is closed.
    """
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from natsort import natsorted

    # Create a temporary Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.lift()
    root.attributes('-topmost', True)  # Bring the dialog to the front

    # Ensure 'exts' is a list
    if isinstance(exts, str):
        exts = [exts]

    # Prepare filetypes for the dialog
    filetypes = [("*.{}".format(ext.upper()),
                 "*.{}".format(ext)) for ext in exts]
    filetypes.append(("All Files", "*.*"))

    if mode == 'open':
        # Open file dialog in 'open' mode
        name = askopenfilename(
            initialdir=default_dir,
            initialfile=default_file,
            parent=root,
            multiple=multiple,
            title=title,
            filetypes=filetypes
        )
        if multiple:
            # If multiple files are selected, ensure the result is a list
            if isinstance(name, str):
                name = [name]
            name = natsorted(name)  # Sort the filenames naturally
    elif mode == 'save':
        # 'multiple' should not be True in save mode
        if multiple:
            raise ValueError(
                "Parameter 'multiple' must be False in 'save' mode.")
        # Open file dialog in 'save' mode
        name = asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_file,
            parent=root,
            title=title,
            filetypes=filetypes
        )
        if name:
            # Append the default extension if not already present
            if not any(name.endswith(".{}".format(ext)) for ext in exts):
                name += '.{}'.format(exts[0])
    else:
        # Invalid mode provided
        raise ValueError(
            "Unknown mode: '{}'. Use 'open' or 'save'.".format(mode))

    # Destroy the temporary root window
    root.destroy()

    if not name:
        # No file was selected
        print("ERROR: No file(s) chosen")
        return None
    else:
        # Return the selected file path(s)
        return name


def string_to_seed(string):
    if not isinstance(string, str):
        warnings.warn(f'input {type(string)=} is not a string, will convert to'
                       ' string representation.')
        string = str(string)
    # Create a SHA-256 hash of the input string
    hash_object = hashlib.sha256(string.encode())
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    # Use modulo to ensure the seed is within the range of valid numpy seeds
    seed = hash_int % (2**32)
    return seed


def hash_array(arr, length=8, dtype=np.int64):
    """
    create a hash for any array by doing a full hash of the hexdigest

    Parameters
    ----------
    arr : np.ndarray
        any type of array.
    length : int, optional
        how many hash characters to return. The default is 8.
    dtype : np.dtype, optional
        which dtype to convert to, can speed up computation massively.
        The default is np.int64.

    Returns
    -------
    str
        sha1 hash of the hex array.

    """
    arr = arr.astype(dtype)
    return hashlib.sha1(arr.flatten("C")).hexdigest()[:length]

def hash_md5(input_string, length=8):
    """make a persistent md5 hash from a string"""
    # Convert input string to bytes
    input_bytes = input_string.encode('utf-8')
    md5_hash = hashlib.md5(input_bytes).hexdigest()
    return md5_hash[:length]


def make_seed(*args):
    """
    Generate a deterministic, high-entropy seed from variable inputs using SHA-256.

    Parameters
    ----------
    *args : any
        Variable length list of inputs (ints, strings, floats) to combine.

    Returns
    -------
    int
        A 32-bit integer suitable for PRNG seeding.
    """
    # Use a delimiter to prevent collisions between (1, 23) and (12, 3)
    data = b"".join(
        f"{type(a).__name__}:{len(s)}:".encode("utf-8") + s.encode("utf-8")
        for a in args
        for s in [str(a)]
    )

    digest = hashlib.sha256(data).hexdigest()

    # Clip to 32-bit unsigned integer range
    return int(digest, 16) % (2**32)


def get_ch_neighbours(ch_name, n=9, return_idx=False,
                      layout_name='Vectorview-all', plot=False):
    """retrieve the n neighbours of a given MEG channel location.
    Count includes the given origin electrode location"""
    layout = mne.channels.read_layout(layout_name)
    positions = {name.replace(' ', ''): pos[:3] for name, pos in zip(
        layout.names, layout.pos, strict=True)}

    Point = namedtuple('Point', 'name x y z')
    ch = Point(ch_name, *positions[ch_name])
    chs = [Point(ch, *pos) for ch, pos in positions.items()]
    chs = [ch for ch in chs if not (('EOG' in ch.name) or ('IO' in ch.name))]

    def dist(p): return (p.x - ch.x)**2 + (p.y - ch.y)**2 + (p.z - ch.z)**2

    chs_sorted = sorted(chs, key=dist)

    chs_out = [ch.name for ch in chs_sorted[:n]]

    ch_as_in_raw = sorted([ch.replace(' ', '') for ch in layout.names])

    if plot:
        layout.plot(picks=[list(positions).index(ch) for ch in chs_out])
    return sorted([ch_as_in_raw.index(ch) for ch in chs_out]) if return_idx else chs_out

def to_long_df(arr, columns=None, value_name='value', **col_labels):
    """
    Convert an N-dimensional NumPy array to a long-format pandas DataFrame.
    E.g. you have probabilities with (trial, timepoint, proba) but need a long
    style dataframe for plotting with seaborn:

        probas = np.random.rand(16, 50, 10)
        timepoints = np.arange(-100, 400, 10)
        df = to_long_df(probas, columns=['trial', 'timepoint', 'proba'],
                        value_name='probability', timepoint=timepoints)

    Only dimensions for which labels are provided are included in the output.
    The array is linearized in Fortran ('F') order to determine the row order.

    Parameters
    ----------
    arr : np.ndarray
        Input N-dimensional array to be reshaped into long format.
    columns : sequence of str, optional
        Names for each dimension of `arr`. If None (default), dimensions are
        named "dim1", "dim2", ..., "dimN". Length must match `arr.ndim`.
    value_name : str, default="value"
        Name for the column containing array values.
    **col_labels : dict of {str: (array-like or dict)}, optional
        For each dimension name in `columns`, specify either:

        - array-like (1-D, length == size of axis):
          Creates a single output column with the same name as the dimension.

        - dict of {str: array-like}:
          Maps output column names to 1-D sequences of labels, each of length
          equal to the axis size. This produces multiple columns derived from
          the same axis.


    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:

        - `value_name`: Flattened values from `arr`.
        - One or more labeled columns derived from `col_labels`.

        Columns appear in the order `[value_name, *labeled_columns]`.
        Dimensions not present in `col_labels` are omitted.
    """

    arr = np.asarray(arr)
    ndim = arr.ndim

    if columns is None:
        columns = [f'dim{i+1}' for i in range(ndim)]
    elif len(columns) != ndim:
        raise ValueError(f"{len(columns)=} must match {arr.ndim=}")

    # Validate kwargs names early
    unknown = set(col_labels).difference(columns)
    if unknown:
        raise KeyError(f"Unknown column(s) in col_labels: {sorted(unknown)}; valid names: {columns}")

    # Fortran-order linearization to match arr.ravel('F')
    n = arr.size
    lin = np.arange(n)
    coords = np.array(np.unravel_index(lin, arr.shape, order='F')).T  # (n, ndim)

    # Assemble output
    out_data = {value_name: arr.ravel('F')}
    out_cols = [value_name]
    used_colnames = set(out_cols)

    for ax, dim_name in enumerate(columns):
        if dim_name=='_' or dim_name is None or dim_name==False:
            continue
        elif dim_name in col_labels:
            spec = col_labels[dim_name]
        else:
            spec = np.arange(arr.shape[ax])

        # Single sequence ? one column named after the dimension
        if not isinstance(spec, dict):
            labels = np.asarray(spec)
            if labels.ndim != 1 or labels.size != arr.shape[ax]:
                raise ValueError(f"Labels for '{dim_name}' must be 1-D of length {arr.shape[ax]} but is {labels.shape=}")
            out_name = dim_name
            if out_name in used_colnames:
                raise ValueError(f"Duplicate output column name: '{out_name}'")
            out_data[out_name] = labels[coords[:, ax]]
            out_cols.append(out_name)
            used_colnames.add(out_name)
            continue

        # Dict ? multiple output columns
        for out_name, labels in spec.items():
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.size != arr.shape[ax]:
                raise ValueError(
                    f"Labels for '{dim_name}.{out_name}' must be 1-D of length {arr.shape[ax]}"
                )
            if out_name in used_colnames:
                raise ValueError(f"Duplicate output column name: '{out_name}'")
            out_data[out_name] = labels[coords[:, ax]]
            out_cols.append(out_name)
            used_colnames.add(out_name)

    return pd.DataFrame(out_data, columns=out_cols)


def low_priority():
    """ Set the priority of the process to below-normal (cross platform).

    subprocesses will inherit the niceness. prevents hogging your CPU"""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(5)


def telegram_callback(on_begin=False, on_finish=False, on_error=True, parse_mode='HTML'):
    """Decorator to notify via telegram_send at begin/finish/error.
    Args:
        on_begin: Send when function starts.
        on_finish: Send when function ends.
        on_error: Send on exception with traceback.
        parse_mode: 'HTML' or 'Markdown'.
    """
    def decorator(func):
        func_name = func.__name__
        mod = inspect.getmodule(func)
        script_path = getattr(mod, '__file__', None) or sys.argv[0] or '<interactive>'
        script_name = Path(script_path).name

        @wraps(func)
        def wrapped(*args, **kwargs):
            t0 = time.time()
            if on_begin:
                msg = (
                    f"<b>{_esc(func_name)}</b> in {_esc(script_name)} began"
                    if parse_mode.upper() == 'HTML'
                    else f"*{func_name}* in {script_name} began"
                )
                _safe_send(msg, parse_mode)

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                if on_error:
                    elapsed = _fmt_duration(time.time() - t0)
                    if parse_mode.upper() == 'HTML':
                        tb = _esc(traceback.format_exc())
                        err = _esc(f"{e.__class__.__name__}: {e}")
                        msg = (
                            f"<b>{_esc(func_name)}</b> in {_esc(script_name)} errored after {_esc(elapsed)}\n"
                            f"<code>{err}</code>\n<code>{tb}</code>"
                        )
                    else:
                        tb = traceback.format_exc()
                        msg = (
                            f"*{func_name}* in {script_name} errored after {elapsed}\n"
                            f"```\n{e.__class__.__name__}: {e}\n{tb}\n```"
                        )
                    _safe_send(msg, parse_mode)
                raise
            else:
                if on_finish:
                    elapsed = _fmt_duration(time.time() - t0)
                    msg = (
                        f"<b>{_esc(func_name)}</b> in {_esc(script_name)} finished after {_esc(elapsed)}"
                        if parse_mode.upper() == 'HTML'
                        else f"*{func_name}* in {script_name} finished after {elapsed}"
                    )
                    _safe_send(msg, parse_mode)
                return result

        return wrapped
    return decorator


def _safe_send(msg, parse_mode):
    """Send message; never raise if telegram_send fails."""
    import telegram_send
    try:
        telegram_send.send(messages=[msg], parse_mode=parse_mode)
    except ModuleNotFoundError as e:
        print('telegram_send not found, please install via pip')
    except Exception:
        try:
            telegram_send.send(messages=[msg])
        except Exception:
            pass


def _fmt_duration(seconds):
    """Return human-readable duration."""
    seconds = float(seconds)
    if seconds < 60:
        s = int(round(seconds))
        return f"{s} second" if s == 1 else f"{s} seconds"
    if seconds < 3600:
        m = int(round(seconds / 60))
        return f"{m} minute" if m == 1 else f"{m} minutes"
    h = seconds / 3600.0
    h_disp = f"{h:.1f}" if h < 10 else f"{int(round(h))}"
    return f"{h_disp} hour" if float(h_disp) == 1.0 else f"{h_disp} hours"
