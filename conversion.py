# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:21:20 2024

@author: Simon Kern (@skjerns)
"""
import mne
import numpy as np
from scipy.stats import zscore

def fif2edf(fif_file, chs=None, edf_file=None):
  """
  Convert a FIF file to an EDF file using pyedflib.

  This function reads an EEG data file in FIF format using MNE-Python and converts it to EDF format using pyedflib.
  Optionally, a subset of channels can be selected for conversion.
  Stimulus events are extracted and included as annotations in the EDF file.

  Parameters
  ----------
  fif_file : str
      Path to the input FIF file.
  chs : list or None, optional
      List of channel indices or names to include in the EDF file. If None,
      a default set of channels is selected automatically. Default is None.
  edf_file : str or None, optional
      Path to the output EDF file. If None, the output filename is the same as
      the input filename with '.edf' appended. Default is None.

  Returns
  -------
  None

  Notes
  -----
  This function uses MNE-Python to read the FIF file and pyedflib to write the EDF file.
  Stimulus events are extracted from the 'STI101' channel and included as annotations.
  Channels are normalized using z-score normalization before writing to the EDF file.

  Examples
  --------
  Convert a FIF file to EDF format:

  >>> fif2edf('sample.fif')

  Specify channels to include:

  >>> fif2edf('sample.fif', chs=[0, 1, 2, 'STI101'])

  Specify output EDF filename:

  >>> fif2edf('sample.fif', edf_file='output.edf')
  """
  from pyedflib import highlevel

  # Read the FIF file using MNE-Python, loading data into memory
  raw = mne.io.read_raw_fif(fif_file, preload=True)

  if chs is None:
      # If no channels are specified, automatically select channels
      n_chs = len(raw.ch_names)
      load_n_channels = 6  # Number of channels to load by default
      if n_chs <= load_n_channels:
          # Load all channels if total channels is less than or equal to load_n_channels
          chs = list(range(n_chs))
      else:
          # Select 'load_n_channels' evenly spaced channels from the first half
          chs = np.unique(np.linspace(0, n_chs // 2 - 2, load_n_channels).astype(int))
      chs = [int(x) for x in chs]  # Ensure channel indices are integers

      # Try to include the 'STI101' channel, typically used for stimuli
      try:
          chs += [raw.ch_names.index("STI101")]
      except ValueError:
          # 'STI101' channel not found; proceed without it
          pass

  if edf_file is None:
      # Set the output EDF filename if not provided
      # Default is input filename with '.edf' appended
      edf_file = fif_file + ".edf"

  # Create annotations from events detected in the 'STI101' stimulation channel
  sfreq = raw.info["sfreq"]  # Sampling frequency
  events = mne.find_events(raw, shortest_event=1, stim_channel="STI101").astype(float).T
  # Adjust event times to be relative to the start of data, in seconds
  events[0] = (events[0] - raw.first_samp) / sfreq
  # Create annotations list: [onset, duration, description]
  annotations = [[s[0], -1 if s[1] == 0 else s[1], str(int(s[2]))] for s in events.T]

  # Alternatively, create annotations directly from 'stim' channel data
  stim = raw.copy().pick("stim").get_data().flatten()
  # Identify times when stim channel is greater than zero (stimulus triggers)
  trigger_times = np.where(stim > 0)[0] / sfreq
  trigger_desc = stim[stim > 0]
  # Find indices where the time difference between triggers is greater than 2 samples
  where_next = [0] + [x for x in np.where(np.diff(trigger_times) > 1 / sfreq * 2)[0]]
  trigger_times = trigger_times[where_next]
  trigger_desc = trigger_desc[where_next]
  # Create annotations for the stimuli
  annotations2 = [
      (t, -1, "STIM " + str(d))
      for t, d in zip(trigger_times, trigger_desc, strict=True)
  ]

  # Select the specified channels from the raw data (modifies 'raw' in-place)
  picks = raw.pick(chs)
  # Get the data for the selected channels
  data = raw.get_data()
  # Normalize the data using z-score normalization along the time axis
  data = zscore(data, axis=1)
  # Replace NaNs with zeros
  data = np.nan_to_num(data)
  # Get the names of the selected channels (after picking)
  ch_names = picks.ch_names

  # Create the EDF file header, specifying the technician name
  header = highlevel.make_header(technician="fif2edf-skjerns")
  # Include the annotations in the header
  header["annotations"] = annotations

  # Create signal headers for each channel
  signal_headers = []
  for name, signal in zip(ch_names, data, strict=True):
      # Determine the physical minimum and maximum values for the signal
      pmin = signal.min()
      pmax = signal.max()
      # Ensure that physical_min and physical_max are not equal
      if pmin == pmax:
          pmin = -1
          pmax = 1
      # Create signal header for the channel
      shead = highlevel.make_signal_header(
          name, sample_rate=sfreq, physical_min=pmin, physical_max=pmax
      )
      signal_headers.append(shead)

  # Write the data to EDF file using pyedflib
  highlevel.write_edf(edf_file, data, signal_headers, header=header)
