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
               case_sensitive=False, as_path=False):
    """List files in a directory matching given extensions or glob patterns.

    Parameters
    ----------
    path : str
        Directory to search in.
    exts : str or list, optional
        Extension(s) to match (e.g. '.jpg' or ['jpg', 'png']). Applied
        additively with `patterns`: files are first matched against
        `patterns`, then filtered down to only those matches whose
        extension is in `exts`. If `patterns` is not given, it defaults
        to '*' so `exts` alone still filters the whole directory.
    patterns : str or list, optional
        Glob pattern(s) supported by pathlib.Path (e.g. '*.txt', 'rfc_*.clf').
        Defaults to '*' when neither `exts` nor `patterns` is given.
    relative : bool, default False
        Return paths relative to `path` instead of absolute.
    recursive : bool, default False
        Also search subfolders (prepends '**/' to each pattern).
    subfolders : bool, optional
        Deprecated alias for `recursive`.
    only_folders : bool, default False
        Return matching directories instead of files.
    max_results : int, optional
        Stop after collecting this many results.
    case_sensitive : bool, default False
        Match patterns case-sensitively.
    as_path : bool, default False
        return pathlib.Path instead of strings

    Returns
    -------
    list of str
        Naturally sorted, de-duplicated file (or folder) paths.
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

    if not patterns:
        # exts alone (or nothing at all) should still search the whole dir
        patterns = ['*']

    # normalize extensions, e.g. '*.jpg' / '.jpg' / 'jpg' -> 'jpg'
    exts = [ext.replace('*', '').lstrip('.').lower() for ext in exts]

    # if recursiveness is asked, prepend the double asterix to each pattern
    if recursive: patterns = ['**/' + pattern for pattern in patterns]

    # collect files for each pattern, then (additively) filter by extension
    files = []
    fcount = 0
    for pattern in patterns:
        if not case_sensitive:
            pattern = insensitive_glob(pattern)
        for filename in p.glob(pattern):
            if filename.is_file():
                if only_folders or filename in files:
                    continue
                if exts and filename.suffix.lstrip('.').lower() not in exts:
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
    files_sorted = sorted(files, key=natsort_key)
    if as_path:
        files_sorted = [Path(p) for p in files_sorted]
    return files_sorted

def get_streaks(arr):
    """helper function to get indices of streaks automatically
    i.e. [1,2,3,4,8,9] -> [[1,4], [8,9]]
    transform dict_values to list and then to array.

    returns: min and max of the array across all dimensions"""
    if len(arr)==0:
        return np.array([])
    arr = np.unique([x for x in arr])
    streaks = np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
    streaks = [(s[0], s[-1]) for s in streaks]
    return np.array(streaks)


def get_clusters(arr, start=0):
    """Get start/end indices of contiguous clusters of same values.

    Parameters
    ----------
    arr : array-like
        1D input array.
    arr : int
        index at which counting should start.

    Returns
    -------
    list of [value, [start, end]]
        Inclusive start/end indices for each cluster.
    """
    if len(arr) == 0:
        return []
    arr = np.asarray(arr)
    change = np.where(np.diff(arr) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [len(arr) - 1]])
    return [[arr[s], [s+start, e+start]] for s, e in zip(starts, ends)]

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

def hash_file(file, method='md5', use_cache=False):
    """returns the hexdigested hash for the binary-read file provided
    for any applicable method thath hashlib offers

    Parameters
    ----------
    file : str or pathlib.Path
        path to the file to hash.
    method : str, optional
        name of any hash algorithm offered by hashlib
        (e.g. 'md5', 'sha1', 'sha256'). The default is 'md5'.
    use_cache : bool, optional
        if True, avoid re-reading the file contents when possible: the hash
        is looked up from a joblib disk cache keyed on the file path plus its
        file_signature(), the quick&dirty fingerprint of its `os.stat`. If
        the fingerprint is unchanged from a previous call, the cached hash is
        returned instead of re-hashing. This can miss a change in the rare
        case content is overwritten while size and both timestamps stay
        identical. The default is False (always re-hash).

    Returns
    -------
    str
        hexdigest of the file contents.
    """
    from joblib import Memory
    memory = Memory(location=str(Path.home() / '.cache' / 'meg_utils' / 'hash_file'),
                    verbose=0)

    def _read_hash():
        hasher = hashlib.new(method)
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    if not use_cache:
        return _read_hash()

    @memory.cache
    def _cached_hash(path, method, signature):
        # signature is unused, it is only here to be part of the cache key
        return _read_hash()

    return _cached_hash(str(Path(file).resolve()), method,
                        file_signature(file))

def file_signature(file):
    """cheap fingerprint of a file, that changes when the file changes

    Uses the same os.stat fields as hash_file(use_cache=True) - size, mtime,
    ctime and inode - but never reads the contents, so there is no md5. It
    therefore only weakly indicates that a file is still the same: it misses
    a change in the rare case that contents are overwritten while size and
    both timestamps stay identical.

    Parameters
    ----------
    file : str | Path | list | tuple | None
        A file path, or a list of them (e.g. several log files belonging to
        one recording). Order is kept, so a reordered list is a new
        signature. None and non-existing files each get their own signature,
        so that a file appearing later does not reuse the result from when it
        was still missing.

    Returns
    -------
    tuple | None
        (path, size, mtime_ns, ctime_ns, inode) for an existing file,
        (path, None) for a missing one, None for None, and a tuple of these
        for a list of files.
    """
    import os
    if file is None:
        return None
    if isinstance(file, (list, tuple)):
        return tuple([file_signature(f) for f in file])
    path = Path(file)
    if not path.exists():
        return (str(path), None)
    stat = os.stat(path)
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns, stat.st_ino)

def cache_on_files(*file_params, memory=None, verbose=0):
    """disk-cache a function, invalidating it when its input *files* change

    joblib.Memory keys its cache on the arguments a function was called with.
    For a function that takes a file *path* that is not enough: the path
    stays the same while the file behind it changes, and the stale result
    would be returned forever. This decorator declares which parameters hold
    file paths. On each call their file_signature() is computed and handed to
    the cached function as an extra argument - unused by the body, but part
    of the cache key, so a changed file misses the cache.

    Can be chained on top of a joblib cache, to pick the Memory and its
    options up from there:

        @cache_on_files('filename')
        @memory.cache
        def func(filename): ...

    Note that this only works in that order (cache_on_files on the outside),
    and that it is not a chain of two caches: the fingerprint has to take
    part in the key that joblib computes, and joblib only ever hashes the
    arguments of the function it wraps itself. So the MemorizedFunc below is
    unwrapped and rebuilt with the same Memory, rather than called through.

    The original function is kept as `.uncached` and the joblib MemorizedFunc
    as `.cached`, so one function's cache can be cleared without touching the
    rest.

    Parameters
    ----------
    *file_params : str, optional
        Names of the parameters that hold a file path or a list of file
        paths. Must be parameters of the decorated function. If none are
        given (or None), every argument is inspected on each call instead,
        and the ones that point at an existing file are fingerprinted. A
        path that does not exist is left alone and simply
        hashed as the string it is, so a file that only appears later starts
        taking part in the key from then on, and the result from when it was
        still missing is not reused.
    memory : joblib.Memory | str | Path, optional
        Where to cache. A Memory is used as it is, a path is turned into one.
        The default is ~/.cache/meg_utils/cache_on_files. Ignored when
        chained below a memory.cache, that Memory is used instead.
    verbose : int
        Verbosity of a Memory that is created here, ignored otherwise.

    Returns
    -------
    callable
        The wrapped function, with `.uncached` (the undecorated function, to
        bypass the cache) and `.cached` (the joblib MemorizedFunc, e.g. for
        `.clear()`) attached to it.

    Examples
    --------
    >>> @cache_on_files('log_file')
    ... def parse_log(log_file, mode='fast'):
    ...     return open(log_file).read()

    a parameter can just as well hold several files, and the cache is
    invalidated if any one of them changes:

    >>> @cache_on_files('recording', 'log_files', memory='/tmp/my-cache')
    ... def check(recording, log_files, strict=True):
    ...     ...
    >>> check.uncached(rec, logs)   # doctest: +SKIP
    >>> check.cached.clear()        # doctest: +SKIP

    without any parameter names, whichever argument happens to be an
    existing file is fingerprinted:

    >>> @cache_on_files()
    ... def check_anything(this, that):
    ...     ...
    """
    from joblib import Memory
    from joblib.memory import MemorizedFunc
    # cache_on_files() and cache_on_files(None) both mean "find them yourself"
    file_params = [param for param in file_params if param is not None]
    if isinstance(memory, Memory):
        pass
    elif memory is None:
        memory = Memory(str(Path.home() / '.cache' / 'meg_utils' /
                            'cache_on_files'), verbose=verbose)
    else:
        memory = Memory(str(memory), verbose=verbose)

    def auto_signature(value):
        """file_signature of value, but only if it is a file to begin with

        Used when no file_params were named. Anything that is not path-like,
        and any path that does not exist (yet), is left alone and ends up
        hashed as the plain string it is. Strings that cannot even be a path
        (too long, null bytes, ...) are answered with None instead of raising.
        """
        import os
        if isinstance(value, (list, tuple)):
            found = tuple([auto_signature(item) for item in value])
            return found if any([s is not None for s in found]) else None
        if not isinstance(value, (str, os.PathLike)):
            return None
        try:
            exists = Path(value).is_file()
        except (OSError, ValueError):
            return None
        return file_signature(value) if exists else None

    def decorator(func):
        func_memory = memory  # a local one, decorator can be reused
        if isinstance(func, MemorizedFunc):
            # chained on top of a memory.cache: take that Memory over, so
            # that the result ends up where the user asked for it
            assert not func.ignore, ('ignore= cannot be passed through '
                                     'cache_on_files, put it on the outside')
            location = Path(func.store_backend.location)
            if location.name == 'joblib':  # Memory appends this itself
                location = location.parent
            func_memory = Memory(str(location), mmap_mode=func.mmap_mode,
                                 compress=func.compress, verbose=verbose)
            func = func.func

        signature = inspect.signature(func)
        unknown = [p for p in file_params if p not in signature.parameters]
        assert not unknown, f'{unknown} are no parameters of {func.__name__}'
        reserved = [p for p in ('_file_signatures', '_source')
                    if p in signature.parameters]
        assert not reserved, f'{reserved} are reserved by cache_on_files'

        # joblib identifies a function by module+qualname, and invalidates the
        # cache when its source changes. Both would point at the wrapper here,
        # so that every decorated function would end up in the same cache
        # directory: @wraps copies the identity of the real function over, and
        # its source is hashed into the key to keep the invalidation.
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):  # e.g. defined in a REPL
            source = f'{func.__module__}.{func.__qualname__}'
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]

        # joblib hashes a positional and a keyword argument differently, so
        # f(x) and f(x=x) would end up as two cache entries. Passing
        # everything by name avoids that (and makes defaults explicit, so
        # f(x) and f(x, mode='text') share an entry as well). Only possible
        # if every parameter *can* be passed by name.
        by_name = all([p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                       for p in signature.parameters.values()])

        # the two extra arguments come first and are positional: joblib
        # cannot map keyword-only parameters of a *args function
        def _cached(_file_signatures, _source, *args, **kwargs):
            # both are ignored here, they only exist to become part of the
            # cache key that joblib computes
            return func(*args, **kwargs)

        # copy the identity over by hand instead of using @wraps: wraps would
        # also set __wrapped__, and joblib follows that to the signature of
        # the original function, which does not take the two extra arguments
        for attr in ('__module__', '__name__', '__qualname__', '__doc__'):
            setattr(_cached, attr, getattr(func, attr, None))

        cached = func_memory.cache(_cached)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if file_params:
                signatures = tuple([file_signature(bound.arguments.get(param))
                                    for param in file_params])
            else:
                # no parameters declared: whatever is an existing file counts
                found = [(name, auto_signature(value))
                         for name, value in bound.arguments.items()]
                signatures = tuple([(name, sig) for name, sig in found
                                    if sig is not None])
            if by_name:
                return cached(signatures, source_hash, **bound.arguments)
            return cached(signatures, source_hash, *args, **kwargs)

        wrapper.uncached = func
        wrapper.cached = cached
        return wrapper
    return decorator

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

def compress_dataframe(df, force_float=None, strings='categorical',
                       categorical_threshold=0.5):
    """
    Compress pandas DataFrame column types to reduce memory usage.

    Applies the following optimizations:
    - Integer columns: downcast to smallest possible int type
    - Float columns: downcast to float32 where possible
    - String/object columns: convert to categorical if beneficial

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to compress.
    force_float : numpy dtype or None, optional
        If specified, force all float columns to this dtype (e.g., np.float32).
        If None (default), automatically downcast to float32 where possible
        using pandas' downcast functionality.
    strings : str or None, optional
        How to handle string/object columns. Options:
        - 'categorical': convert to categorical if beneficial (default)
        - None: skip string compression
    categorical_threshold : float, optional
        Maximum ratio of unique values to total values for a string column
        to be converted to categorical. Default is 0.5 (50%).
        E.g., if a column has 100 rows and 40 unique values (40%),
        it will be converted to categorical.

    Returns
    -------
    pandas.DataFrame
        DataFrame with compressed column types.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5],
    ...                    'c': ['x', 'x', 'y']})
    >>> df_compressed = compress_dataframe(df)
    >>> df_compressed.dtypes
    a       int8
    b    float32
    c   category
    dtype: object
    """
    df = df.copy()

    for col in df.columns:
        col_dtype = df[col].dtype

        # Handle integer columns
        if np.issubdtype(col_dtype, np.integer):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # Handle float columns
        elif np.issubdtype(col_dtype, np.floating):
            if force_float is not None:
                df[col] = df[col].astype(force_float)
            else:
                # Downcast to smallest float type (float32 minimum)
                df[col] = pd.to_numeric(df[col], downcast='float')

        # Handle string/object columns
        elif col_dtype == 'object' or col_dtype.name == 'string':
            if strings == 'categorical':
                n_unique = df[col].nunique()
                n_total = len(df[col])
                # Convert to categorical if ratio of unique values is below threshold
                # and there's at least some data
                if n_total > 0 and (n_unique / n_total) <= categorical_threshold:
                    df[col] = df[col].astype('category')

    return df


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


def long_df_to_array(df, value_name, columns, fill_value=np.nan):
    """
    Convert a long-format DataFrame to an N-dimensional NumPy array.

    Inverse of :func:`to_long_df`. For each combination of values across the
    given dimension columns, the corresponding value is placed in the array at
    the matching multi-dimensional index.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format DataFrame, e.g. as produced by :func:`to_long_df`.
    columns : list of str
        Names of the DataFrame columns that correspond to array dimensions,
        given in the desired dimension order. The unique sorted values in each
        column define the axis labels and the array shape along that axis.
    value_name : str, default='value'
        Name of the column whose values are placed into the array.
    fill_value : scalar, default=np.nan
        Value inserted for index combinations not present in ``df``.

    Returns
    -------
    np.ndarray
        N-dimensional array with
        ``shape = (len(unique(col)) for col in columns)``.

    Examples
    --------
    >>> probas = np.random.rand(16, 50, 10)
    >>> timepoints = np.arange(-100, 400, 10)
    >>> df = to_long_df(probas, columns=['trial', 'timepoint', 'proba'],
    ...                 value_name='probability', timepoint=timepoints)
    >>> arr = long_df_to_array(df, columns=['trial', 'timepoint', 'proba'],
    ...                        value_name='probability')
    >>> arr.shape
    (16, 50, 10)
    >>> np.allclose(arr, probas)
    True
    """
    # --- Input validation ---
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in DataFrame. "
                       f"Available: {list(df.columns)}")
    if value_name not in df.columns:
        raise KeyError(f"Value column '{value_name}' not found in DataFrame. "
                       f"Available: {list(df.columns)}")
    if value_name in columns:
        raise ValueError(f"'{value_name}' appears in both `value_name` and "
                         f"`columns` — these must be disjoint.")

    # Warn about NaN in dimension columns (searchsorted gives wrong indices)
    for col in columns:
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values. Dimension columns must "
                f"not contain NaN as this produces incorrect array indices.")

    # Sorted unique values per dimension → axis labels and shape
    uniques = [np.sort(df[col].unique()) for col in columns]
    shape = tuple(len(u) for u in uniques)

    # Map each dimension column to 0-based integer indices via searchsorted
    indices = tuple(
        np.searchsorted(uniques[ax], df[columns[ax]].values)
        for ax in range(len(columns))
    )

    # Check that each combination of indices is unique (no duplicate rows)
    multi_idx = np.column_stack(indices)
    n_rows, n_unique = len(multi_idx), len(np.unique(multi_idx, axis=0))
    if n_unique != n_rows:
        raise ValueError(
            f"Duplicate index combinations found: {n_rows} rows but only "
            f"{n_unique} unique combinations across columns {columns}. "
            f"Each combination must map to exactly one value."
        )

    # Allocate output and scatter values
    arr = np.full(shape, fill_value)
    arr[indices] = df[value_name].values

    return arr


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


def convert_to_numeric(df, convert_dtypes=True, inplace=True):
    """
    Convert DataFrame columns to numeric dtypes where possible.

    Tries ``pd.to_numeric`` on every non-numeric column. If all values in a
    column convert successfully, the numeric column is kept; otherwise the
    original column is retained unchanged. Columns that contain any
    non-numeric value (including ``None`` / ``NaN``) will not be converted.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    convert_dtypes : bool, default=True
        If True, call ``df.convert_dtypes()`` at the end to further optimize
        dtypes (e.g. nullable integer types, string dtype).
    inplace : bool, default=True
        If True, modify *df* in place. If False, operate on a copy.

    Returns
    -------
    pandas.DataFrame
        DataFrame with eligible columns converted to numeric types.
    """
    if not inplace:
        df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            df[col] = converted
        except (ValueError, TypeError):
            pass
    if convert_dtypes:
        df = df.convert_dtypes()
    return df


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
