# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:26 2024

@author: Simon Kern (@skjerns)
"""
from collections import namedtuple
import warnings
import mne
import hashlib
import numpy as np
import json

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
