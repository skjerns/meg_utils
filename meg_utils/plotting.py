# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:18:57 2024

@author: Simon Kern (@skjerns)
"""
import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image, PngImagePlugin
import json
import inspect
import subprocess

def _infer_layout(n_values):
    louts_dir = f'{mne.__path__[0]}/channels/data/layouts/'
    louts = [f for f in os.listdir(louts_dir) if f.endswith('.lout')]

    for lout in louts:
        layout = mne.channels.read_layout(os.path.join(louts_dir, lout))
        if len(layout.pos) == n_values:
            print(f'assuming layout = {lout}')
            return layout
    raise ValueError(f'No layout found with {n_values} positions in {louts}')

def plot_sensors(
    values,
    layout='auto',
    positions=None,
    title="Sensors active",
    mode="size",
    color=None,
    ax=None,
    vmin=None,
    vmax=None,
    cmap="Reds",
    **kwargs,
):
    """
    Plot sensor positions with markers representing various data values.

    This function visualizes sensor data on a predefined sensor layout using different
    visualization modes. It supports sizing markers based on values, binary
    activation, multi-class binary activation, and percentage-based activation.

    Parameters
    ----------
    values : array-like
        The data values to visualize on the sensors. The shape depends on the selected
        mode:
        - For `mode="size"` and `mode="binary"`, `values` should be a 1D array with
          length equal to the number of sensors.
        - For `mode="multi_binary"` and `mode="percentage"`, `values` should be a 2D
          array where each row represents a different class or condition.

    layout: str
        name of the layout found int /mne/channels/data/layouts/*.lout
        e.g. Vectorview-all. If 'auto' will try to match the number of
        values to a layout that has the corresponding number of channels

    title : str, optional
        The title of the plot. Default is "Sensors active".

    mode : {"size", "binary", "multi_binary", "percentage"}, optional
        The visualization mode determining how the sensor data is represented:

        - `"size"`: Marker sizes are scaled according to the `values`.
        - `"binary"`: Sensors are displayed as active or inactive based on `values`.
        - `"multi_binary"`: Multiple binary classes are visualized with different colors.
        - `"percentage"`: Percentage of activations per sensor is visualized.

        Default is `"size"`.

    color : str or array-like, optional
        The color to use for markers in `"binary"` mode. If `None`, defaults to
        `"red"`. Ignored in other modes.

    ax : matplotlib.axes.Axes, optional
        The matplotlib Axes object to draw the plot on. If `None`, a new Axes is
        created. Default is `None`.

    vmin : float, optional
        The minimum data value for colormap scaling in `"size"` and `"percentage"`
        modes. If `None`, uses the minimum of `values`. Default is `None`.

    vmax : float, optional
        The maximum data value for colormap scaling in `"size"` and `"percentage"`
        modes. If `None`, uses the maximum of `values`. Default is `None`.

    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use for representing data values in `"size"` and
        `"percentage"` modes. Default is `"Reds"`.

    **kwargs
        Additional keyword arguments passed to the underlying plotting functions.
        For example, in `"multi_binary"` mode, extra keyword arguments are passed to
        `seaborn.scatterplot`.

    Returns
    -------
    matplotlib.scatter.PathCollection or matplotlib.legend.Legend or None
        - In `"size"` mode, returns the scatter plot object.
        - In `"multi_binary"` mode, returns the seaborn scatter plot object.
        - In `"percentage"` mode, returns the legend object.
        - In other modes, returns `None`.

    Raises
    ------
    ValueError
        If an unsupported mode is provided.

    Notes
    -----
    - The sensor layout is based on the "Vectorview-all" layout from MNE.
    - Additional graphical elements such as circles and polygons are added for
      orientation (e.g., eyes and nose).

    Examples (created by o1)
    --------
    >>> import numpy as np
    >>> # Example data for size mode
    >>> sensor_values = np.random.rand(306)
    >>> plot_sensors(sensor_values, mode="size", cmap="viridis")

    >>> # Example data for binary mode
    >>> binary_values = np.random.randint(0, 2, size=306)
    >>> plot_sensors(binary_values, mode="binary", color="blue")

    >>> # Example data for multi_binary mode
    >>> multi_binary_values = np.random.randint(0, 2, size=(3, 306))
    >>> plot_sensors(multi_binary_values, mode="multi_binary")

    >>> # Example data for percentage mode
    >>> percentage_values = np.random.randint(0, 2, size=(100, 306))
    >>> plot_sensors(percentage_values, mode="percentage")
    """
    if positions is None and layout:
        if layout=='auto':
            layout = _infer_layout(len(values))
        else:
            layout = mne.channels.read_layout(layout)
        positions = layout.pos[:, :2].T
    elif positions is not None and layout not in [None, False]:
        warnings.warn(f'positions has been provided, {layout=} will be ignored')

    def jitter(values):
        values = np.array(values)
        return values * np.random.normal(1, 0.01, values.shape)

    if ax is None:
        fig = plt.figure(figsize=[7, 7], constrained_layout=False)
        ax = plt.gca()
    else:
        fig = ax.figure
    plot = None
    ax.clear()
    if mode == "size":
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)

        scaling = (fig.get_size_inches()[-1] * fig.dpi) / 20
        sizes = scaling * (values - np.min(values)) / vmax
        plot = ax.scatter(
            *positions, s=sizes, c=values, vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.75
        )

    elif mode == "binary":
        assert values.ndim == 1
        if color is None:
            color = "red"
        pos_true = positions[:, values > 0]
        pos_false = positions[:, values == 0]
        ax.scatter(*pos_true, marker="o", color=color)
        ax.scatter(*pos_false, marker=".", color="black")

    elif mode == "multi_binary":
        assert values.ndim == 2
        x = []
        y = []
        classes = []
        pos_false = positions[:, values.sum(0) == 0]
        ax.scatter(*pos_false, marker=".", color="black", alpha=0.5, s=5)
        for i, value in enumerate(values):
            pos_true = positions[:, values[i] > 0]
            x.extend(pos_true[0])
            y.extend(pos_true[1])
            classes.extend([f"class {i}"] * len(pos_true[0]))
        data = pd.DataFrame({"x": jitter(x), "y": jitter(y), "class": classes})
        sns.scatterplot(data=data, x="x", y="y", hue="class", ax=ax, **kwargs)

    elif mode == "percentage":
        assert values.ndim == 2
        perc = (values > 0).mean(0)
        pos_true = positions[:, perc > 0]
        pos_false = positions[:, perc == 0]
        sc1 = ax.scatter(
            *pos_true, marker="o", c=perc[perc > 0], cmap="gnuplot_r", vmin=0.05
        )
        ax.scatter(*pos_false, marker=".", color="black",
                   alpha=0.5, s=5, **kwargs)
        labels = [f"beta shared by {x}" for x in np.unique((values > 0).sum(0))[
            1:]]
        fig.legend(handles=sc1.legend_elements()[0], labels=labels)

    else:
        raise ValueError(
            'Mode must be "size","binary", "multi_binary", "percentage"')

    # add lines for eyes and nose for orientation
    ax.add_patch(plt.Circle((0.475, 0.475), 0.475, color="black", fill=False))
    ax.add_patch(plt.Circle((0.25, 0.85), 0.04, color="black", fill=False))
    ax.add_patch(plt.Circle((0.7, 0.85), 0.04, color="black", fill=False))
    ax.add_patch(
        plt.Polygon(
            [[0.425, 0.9], [0.475, 0.95], [0.525, 0.9]], fill=False, color="black"
        )
    )
    ax.set_axis_off()
    ax.set_title(title)
    return plot

def make_sensor_importance_gif(output_filename, data_x=None, data_y=None, importances=None,
                               accuracies = None, layout='auto', tmin=None, tmax=None, n_jobs=-1, n_folds=10,
                               fps=0.2):
    from meg_utils import decoding
    # Compute timepoints

    if tmin is None or tmax is None:
        timesteps = np.arange(data_x.shape[-1]) * 10 - 200
    else:
        timesteps = np.linspace(tmin*1000, tmax*1000, data_x.shape[-1], dtype=int)

    assert (data_x is None) == (data_y is None), 'supply data_x or data_y or importances'
    assert (data_x is None) or (importances is None), 'supply either data_x or importances'

    # accuracies = None
    # Compute importances for each timepoint
    if importances is None:
        results = Parallel(n_jobs=n_jobs)(
            delayed(decoding.get_channel_importances)(data_x[:, :, t], data_y, n_folds=n_folds)
            for t in range(data_x.shape[-1])
        )

        accuracies = np.mean([r for r, _ in results], -1)  # accuracies
        importances = [i for _, i in results]  # importances

    vmin = np.min(importances)
    vmax = np.max(importances)

    image_buffers = []

    if accuracies is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        mosaic = '\n'.join(['AAAA',
                  'AAAA',
                  'AAAA',
                  'BBBB'])
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(6, 7))
        ax = axs['A']
        ax_acc = axs['B']
        ax_acc.plot(timesteps, accuracies)
        ax_acc.set_xlabel('time after stim onset')
        ax_acc.vlines(0, *ax_acc.get_ylim(), color='black')
        ax2 = ax_acc.twinx()


    for tp_idx, tp in enumerate(timesteps):
        imp = importances[tp_idx]

        ax.clear()
        ax.set_axis_off()
        plot_sensors(imp, layout=layout, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(f'Sensor importance @ {tp} ms')

        if accuracies is not None:
            ax2.clear()
            ax2.vlines(tp, *ax2.get_ylim(), color='red')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_buffers.append(imageio.v3.imread(buf))

    # Save GIF
    imageio.mimsave(output_filename, image_buffers, fps=fps)


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True,
                  **kwargs):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     fill=False, linewidth=1, **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches



def make_fig(
    n_axs=30,
    bottom_plots=2,
    no_ticks=False,
    suptitle="",
    xlabel="Timepoint",
    ylabel="",
    figsize=None,
    despine=True,
):
    """
    helper function to create a grid space with RxC rows and a
    large row with two axis on the bottom

    returns: fig, axs(size=(rows*columns)), ax_left_bottom, ax_right_bottom
    """

    COL_MULT = 10  # to accomodate also too large axis
    # some heuristic for finding optimal rows and columns
    for columns in [2, 4, 6, 8]:
        rows = np.ceil(n_axs / columns).astype(int)
        if columns >= rows:
            break
    assert columns * rows >= n_axs

    if isinstance(bottom_plots, int):
        bottom_plots = [1 for _ in range(bottom_plots)]
    n_bottom = len(bottom_plots)
    COL_MULT = 1
    if n_bottom > 0:
        for COL_MULT in range(1, 12):
            if (columns * COL_MULT) % n_bottom == 0:
                break
        if not (columns * COL_MULT) % n_bottom == 0:
            warnings.warn(
                f"{columns} cols cannot be evenly divided by {bottom_plots} bottom plots"
            )
    fig = plt.figure(dpi=75, constrained_layout=True, figsize=figsize)
    # assuming maximum 30 participants
    gs = fig.add_gridspec(
        (rows + 2 * (n_bottom > 0)), columns * COL_MULT
    )  # two more for larger summary plots
    axs = []

    # first the individual plot axis for each participant
    for x in range(rows):
        for y in range(columns):
            ax = fig.add_subplot(gs[x, y * COL_MULT : (y + 1) * COL_MULT])
            if no_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            axs.append(ax)
            if len(axs)>=n_axs:
                break
        if len(axs)>=n_axs:
            break
    fig.suptitle(suptitle)

    if len(bottom_plots) == 0:
        return fig, axs

    # second the two graphs with all data combined/meaned
    axs_bottom = []
    step = np.ceil(columns * COL_MULT // n_bottom).astype(int)
    for b, i in enumerate(range(0, columns * COL_MULT, step)):
        if bottom_plots[b] == 0:
            continue  # do not draw* this plot
        ax_bottom = fig.add_subplot(gs[rows:, i : (i + step)])
        if xlabel:
            ax_bottom.set_xlabel(xlabel)
        if ylabel:
            ax_bottom.set_ylabel(ylabel)
        if i > 0 and no_ticks:  # remove yticks on righter plots
            ax_bottom.set_yticks([])
        axs_bottom.append(ax_bottom)
    if despine:
        sns.despine(fig)
    return fig, axs, *axs_bottom

def savefig(fig, file, tight=True, despine=True, metadata=None,
            save_vector=True, **kwargs):
    """
    Save a Matplotlib figure to a specified file with optional adjustments and metadata.

    This function refreshes the figure, applies optional layout adjustments
    (tight layout and despine), and saves the figure to the specified file.
    It ensures the output directory exists and appends a default file extension
    if none is provided. For PNG and JPG files, custom metadata can be embedded.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure to save.
    file : str
        The file path where the figure will be saved. If the file does not
        have an extension, '.png' will be appended.
    tight : bool, optional
        If True, applies `fig.tight_layout()` to adjust the layout of the figure
        before saving. Default is True.
    despine : bool, optional
        If True, removes the top and right spines from the figure using
        `sns.despine()`. Default is True.
    save_vector : bool, optional
        If True, saves additional SVG and PDF copies of the figure to a
        'vectors/' subfolder in the same directory as `file`. The `dpi`
        kwarg is excluded when saving vector formats. Default is True.
    metadata : dict | str | False | None, optional
        Metadata to embed in the image file:
        - dict: Custom metadata key-value pairs
        - str: Single metadata string (stored as 'metadata' key)
        - False: No metadata is added
        - None (default): Auto-generate metadata with script path and git commit hash
        For PNG files, each key-value pair is stored as a text chunk.
        For JPG files, the metadata is JSON-encoded and stored in the comment field.
    **kwargs : dict, optional
        Additional keyword arguments passed to `fig.savefig()`.

    Notes
    -----
    - The function ensures the output directory exists by creating it if necessary.
    - Supported file extensions include 'png', 'jpg', 'svg', and 'pdf'. If no
      extension is provided, '.png' is used by default.
    - For PNG: metadata keys and values are stored as text chunks
    - For JPG: metadata is JSON-encoded and stored as a JPEG comment
    - For SVG: metadata is JSON-encoded in the Dublin Core Description element
    - For PDF: metadata is JSON-encoded in the Keywords field of the PDF info dict

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> metadata = {'author': 'John Doe', 'experiment': 'MEG_2024', 'notes': 'Test data'}
    >>> savefig(fig, 'plot.png', metadata=metadata)
    """
    fig.canvas.draw_idle()   # Refresh only fig1
    fig.canvas.flush_events()  # Process GUI events for fig1
    if despine:
        sns.despine(fig)
    if tight:
        fig.tight_layout()
    fig.canvas.draw_idle()   # Refresh only fig1
    fig.canvas.flush_events()  # Process GUI events for fig1
    fig.show()

    # Ensure the output directory exists
    out_dir = os.path.dirname(file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not file.endswith(('png', 'jpg', 'svg', 'pdf')):
        file = file + '.png'
    fig.savefig(file, **kwargs)

    # Resolve metadata to a dict (or None if disabled)
    if metadata is False:
        resolved_metadata = None
    elif metadata is None:
        resolved_metadata = _generate_default_metadata()
    elif isinstance(metadata, str):
        resolved_metadata = {'metadata': metadata}
    else:
        resolved_metadata = metadata

    if save_vector:
        vec_dir = os.path.join(out_dir, 'vectors') if out_dir else 'vectors'
        os.makedirs(vec_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(file))[0]
        vec_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
        for ext in ('svg', 'pdf'):
            vec_file = os.path.join(vec_dir, f'{basename}.{ext}')
            ext_kwargs = dict(vec_kwargs)
            if resolved_metadata:
                ext_kwargs['metadata'] = _metadata_for_vector(
                    resolved_metadata, ext)
            fig.savefig(vec_file, **ext_kwargs)

    # Add metadata to PNG or JPG files if provided
    if resolved_metadata:
        _add_image_metadata(file, resolved_metadata)



def _metadata_for_vector(metadata, fmt):
    """Convert a metadata dict into the format accepted by Matplotlib's
    SVG/PDF backends.

    For **PDF**, the custom metadata is JSON-encoded into the ``Keywords``
    field of the PDF info dictionary.  For **SVG**, it is stored in the
    Dublin-Core ``Description`` element.
    """
    meta_json = json.dumps(metadata)
    title = metadata.get('script_path', '')
    if fmt == 'pdf':
        return {'Title': title, 'Keywords': meta_json}
    else:  # svg
        return {'Title': title, 'Description': meta_json}


def _generate_default_metadata():
    """
    Generate default metadata including script path and git commit hashes.

    Returns
    -------
    dict
        Dictionary with 'script_path', 'script_git_commit', and 'repo_git_commit' keys.
    """
    metadata = {}

    # Get the calling script path
    caller_path = None
    try:
        frame = inspect.currentframe()
        # Go up the call stack to find the caller outside this module
        caller_frame = frame
        while caller_frame is not None:
            caller_info = inspect.getframeinfo(caller_frame)
            caller_path = caller_info.filename
            # Skip frames from this module
            if not caller_path.endswith('plotting.py'):
                metadata['script_path'] = os.path.abspath(caller_path)
                break
            caller_frame = caller_frame.f_back
    except Exception:
        metadata['script_path'] = 'n/a'
        caller_path = None

    # Get the git commit hash of the caller script
    if caller_path and os.path.exists(caller_path):
        try:
            script_dir = os.path.dirname(os.path.abspath(caller_path))
            script_name = os.path.basename(caller_path)
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H', script_name],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=2,
                check=True
            )
            metadata['script_git_commit'] = result.stdout.strip() or 'n/a'
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            metadata['script_git_commit'] = 'n/a'
    else:
        metadata['script_git_commit'] = 'n/a'

    # Get the git commit hash of the current repository
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2,
            check=True
        )
        metadata['repo_git_commit'] = result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        metadata['repo_git_commit'] = 'n/a'

    return metadata


def _add_image_metadata(filepath, metadata):
    """
    Add metadata to PNG or JPG image files.

    Parameters
    ----------
    filepath : str
        Path to the image file.
    metadata : dict
        Dictionary of metadata key-value pairs to embed in the image.

    Notes
    -----
    - For PNG files: metadata is stored as text chunks
    - For JPG files: metadata is JSON-encoded and stored as a comment
    - All metadata values are converted to strings
    """
    if not isinstance(metadata, dict) or not metadata:
        return

    # Convert all metadata values to strings
    metadata_str = {k: str(v) for k, v in metadata.items()}

    try:
        img = Image.open(filepath)
        img_format = img.format

        if img_format == 'PNG':
            # For PNG: add metadata as text chunks
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata_str.items():
                pnginfo.add_text(key, value)
            img.save(filepath, "PNG", pnginfo=pnginfo)

        elif img_format in ('JPEG', 'JPG'):
            # For JPEG: store metadata as JSON in comment field
            comment = json.dumps(metadata_str)
            # Save with comment - PIL supports the 'comment' parameter for JPEG
            img.save(filepath, "JPEG", comment=comment, quality=95)

        img.close()

    except Exception as e:
        warnings.warn(f"Could not add metadata to {filepath}: {e}")


def label_panels(axs, labels=None, fontsize=14, x=-0.05, y=1.05, **kwargs):
    """
    Add bold letter annotations (e.g. A, B, C) to subplot axes.

    Useful for labelling panels in a figure. Labels are placed in the upper-left
    corner, slightly outside the axes frame.

    Parameters
    ----------
    axs : sequence of matplotlib.axes.Axes
        Axes to annotate. Can be a flat list or numpy array (e.g. from plt.subplots()).
    labels : sequence of str or None, optional
        Labels to place on each axis. If None, uses uppercase letters A, B, C, ...
        Must be at least as long as axs.
    fontsize : int or float, optional
        Font size of the annotation. Default is 14.
    x : float, optional
        Horizontal position in axes-fraction coordinates. Default is -0.05
        (slightly left of the left edge).
    y : float, optional
        Vertical position in axes-fraction coordinates. Default is 1.05
        (slightly above the top edge).
    **kwargs
        Additional keyword arguments passed to ax.text().

    Examples
    --------
    >>> fig, axs = plt.subplots(2, 3)
    >>> annotate_subplots(axs[0])           # label only top row: A B C
    >>> annotate_subplots(axs.flat)         # label all 6 subplots: A B C D E F
    >>> annotate_subplots([ax1, ax3], labels=['A', 'C'])
    """
    import string

    if hasattr(axs, 'flat'):
        axs = list(axs.flat)
    else:
        axs = list(axs)

    if labels is None:
        labels = list(string.ascii_uppercase)

    text_kwargs = dict(fontsize=fontsize, fontweight='bold', va='bottom', ha='left')
    text_kwargs.update(kwargs)

    for ax, label in zip(axs, labels):
        ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


def normalize_lims(axs, which='xy'):
    """
    Synchronize axis and/or color (clim) limits across a collection of Matplotlib Axes.

    Parameters
    ----------
    axs : sequence of matplotlib.axes.Axes
        Axes to normalize. Can be a single Axes, list/tuple, or array (e.g., from plt.subplots()).
    which : {'x','y','v','xy','xv','yv','xyv','both','all'}, default 'xy'
        Characters indicate which limits to synchronize:
            'x'  -> xlim
            'y'  -> ylim
            'v'  -> color limits (clim) of the most recently added image on each Axes.
            'z'  -> synonym of v.
            'c'  -> synonym of v.
        Combinations are allowed by concatenation (e.g., 'xy', 'xv', 'yv', 'xyv').
        Back-compat: 'both' == 'xy'.
        Convenience: 'all'  == 'xyv'.

    Notes
    -----
    * For 'v', only the newest image in each Axes (`ax.images[-1]`) is considered/updated.
    * Axes without images are ignored for the global color range calculation.
    * When possible, the image's underlying array is inspected (np.nanmin / np.nanmax).
      If that fails, the current clim from the image is used as a fallback.
    """
    # Flatten / normalize the axes input to a simple list.
    if hasattr(axs, 'flat'):  # numpy array of Axes
        axs = [ax for ax in axs.flat]

    for axis in which:
        if not axis in 'xyzvc':
            raise ValueError(f'Unknown {axis=} in parameter which, only allowed are xyzvc, with v==z')

    # z is synonym with v
    which = which.replace('z', 'v')
    which = which.replace('c', 'v')

    spec = which.lower()
    if spec == 'both':
        spec = 'xy'
    elif spec == 'all':
        spec = 'xyv'

    # canonical order: x, y, v
    spec = ''.join(ch for ch in 'xyv' if ch in spec)

    # --- X limits ---
    if 'x' in spec:
        xlims = [ax.get_xlim() for ax in axs]
        xmin = min(l[0] for l in xlims)
        xmax = max(l[1] for l in xlims)
        for ax in axs:
            ax.set_xlim(xmin, xmax)

    # --- Y limits ---
    if 'y' in spec:
        ylims = [ax.get_ylim() for ax in axs]
        ymin = min(l[0] for l in ylims)
        ymax = max(l[1] for l in ylims)
        for ax in axs:
            ax.set_ylim(ymin, ymax)

    # --- Color limits (v) ---
    if 'v' in spec:
        # Gather all scalar-mappables and their data-driven mins/maxs
        def mappables(ax):
            items = []
            items.extend(ax.images)  # imshow, matshow
            # pcolormesh/quadmesh, scatter with array, etc.
            items.extend([c for c in ax.collections if hasattr(c, 'get_array') and c.get_array() is not None])
            # contourf returns a ContourSet; treat its collections together via its mappable API if present
            # Many ContourSets store a ScalarMappable-like norm and array on the first collection.
            return items

        vmins, vmaxs = [], []
        per_ax_mappables = []
        for ax in axs:
            mapps = mappables(ax)
            per_ax_mappables.append(mapps)
            for m in mapps:
                try:
                    arr = np.asarray(m.get_array())
                    vmins.append(np.nanmin(arr))
                    vmaxs.append(np.nanmax(arr))
                except Exception:
                    try:
                        v0, v1 = m.get_clim()
                        vmins.append(v0); vmaxs.append(v1)
                    except Exception:
                        pass

        if vmins:
            vmin = np.nanmin(vmins)
            vmax = np.nanmax(vmaxs)
            # Avoid degenerate range
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                return
            for ax, mapps in zip(axs, per_ax_mappables):
                for m in mapps:
                    # Works for Normalize/LogNorm; BoundaryNorm may need more bespoke handling
                    m.set_clim(vmin, vmax)
                    # m.changed() is called by set_clim internally; keeps colorbars in sync
    return


def tornadoplot(data, x=None, y=None, center=0, low_colour='#4c72b0',
                high_colour='#dd8452', ax=None, orient='h',
                low_label='Low', high_label='High',
                sort=True, **kwargs):
    """
    Create a tornado (diverging bar) chart from a DataFrame.

    Bars extend left/right (or down/up) from a *center* value.
    Values below *center* are coloured with *low_colour*, values above with
    *high_colour*.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data.
    x : str
        Column name for the numeric values (bar lengths).
    y : str
        Column name for the category labels (bar labels).
    center : float, optional
        The reference value that separates "low" from "high". Default is 0.
    low_colour : str, optional
        Colour for bars whose value is below *center*. Default is ``'#4c72b0'``.
    high_colour : str, optional
        Colour for bars whose value is >= *center*. Default is ``'#dd8452'``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If None a new figure and axes are created.
    orient : {'h', 'v'}, optional
        ``'h'`` for horizontal bars (default), ``'v'`` for vertical bars.
    low_label : str, optional
        Legend label for bars below *center*. Default is ``'Low'``.
    high_label : str, optional
        Legend label for bars at or above *center*. Default is ``'High'``.
    sort : bool, optional
        If True (default), rows are sorted by value so the largest bars appear
        at the top (horizontal) or right (vertical).
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.barplot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes with the tornado plot.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'feature': list('ABCDE'),
    ...                     'value': [-0.3, 0.5, -0.1, 0.8, 0.2]})
    >>> tornadoplot(df, x='value', y='feature', center=0)
    """
    df = data.copy()

    if x is None or y is None:
        raise ValueError("Both `x` and `y` must be specified.")

    if sort:
        df = df.sort_values(x, ascending=True).reset_index(drop=True)

    colours = [high_colour if v >= center else low_colour for v in df[x]]

    if ax is None:
        _, ax = plt.subplots()

    # assign a unique hue per row so seaborn accepts per-bar colours
    df['_hue'] = range(len(df))
    palette_map = dict(enumerate(colours))

    if orient == 'h':
        sns.barplot(data=df, x=x, y=y, hue='_hue', palette=palette_map,
                    orient='h', legend=False, ax=ax, **kwargs)
        ax.axvline(center, color='black', linewidth=0.8)
    else:
        sns.barplot(data=df, x=y, y=x, hue='_hue', palette=palette_map,
                    orient='v', legend=False, ax=ax, **kwargs)
        ax.axhline(center, color='black', linewidth=0.8)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=high_colour, label=high_label),
               Patch(facecolor=low_colour, label=low_label)]
    ax.legend(handles=handles)

    return ax


def highlight_cells(mask, ax, color='r', linewidth=1, linestyle='solid'):
    """
    Draws borders around the true entries of the mask array on a heatmap plot.

    Parameters:
    - mask (np.ndarray): A 2D binary mask array where True (or 1) entries indicate
                         the cells that should be highlighted with a border.
    - ax (matplotlib.axes.Axes): The axes on which the heatmap is plotted.
    - color (str): Color of the border lines.
    - linewidth (float): Width of the border lines.
    - linestyle (str): Line style of the border lines.
    """
    # Ensure the mask is a 2D array
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D array.")

    # # Check if the mask dimensions match the plotted image dimensions
    # image_shape = ax.images[0].get_array().shape
    # if mask.shape != image_shape:
    #     raise ValueError(f"Mask dimensions {mask.shape} do not match the plotted image dimensions {image_shape}.")

    # Function to check if a cell is outside the mask or out of bounds
    def is_outside_mask(i, j, mask):
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return True
        return not mask[i, j]

    # Loop through each cell in the mask to draw borders around masked regions
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                # Draw top border if the cell above is outside the mask
                if is_outside_mask(i - 1, j, mask):
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw bottom border if the cell below is outside the mask
                if is_outside_mask(i + 1, j, mask):
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw left border if the cell to the left is outside the mask
                if is_outside_mask(i, j - 1, mask):
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)
                # Draw right border if the cell to the right is outside the mask
                if is_outside_mask(i, j + 1, mask):
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, linewidth=linewidth, linestyle=linestyle)


"""
Complete
one-liner
 visualiser for alpha-band travelling waves in MEG.

Just call

    show_alpha_dynamics(raw)

where `raw` is an mne.io.Raw object that has already been
cleaned (Maxwell filter & ICA, etc.).  Nothing else is needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert
from scipy.spatial.distance import cdist


#
