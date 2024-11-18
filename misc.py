# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:26 2024

@author: Simon Kern (@skjerns)
"""
from collections import namedtuple
import mne
import hashlib
import numpy as np

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


def get_ch_neighbours(ch_name, n=9, return_idx=False,
                      layout_name='Vectorview-all', plot=False):
    """retrieve the n neighbours of a given MEG channel location.
    Count includes the given origin electrode location"""
    layout = mne.channels.read_layout(layout_name)
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
