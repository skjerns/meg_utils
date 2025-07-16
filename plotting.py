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

def savefig(fig, file, tight=True, despine=True, **kwargs):
    """
    Save a Matplotlib figure to a specified file with optional adjustments.

    This function refreshes the figure, applies optional layout adjustments
    (tight layout and despine), and saves the figure to the specified file.
    It ensures the output directory exists and appends a default file extension
    if none is provided.

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
    **kwargs : dict, optional
        Additional keyword arguments passed to `fig.savefig()`.

    Notes
    -----
    - The function ensures the output directory exists by creating it if necessary.
    - Supported file extensions include 'png', 'jpg', 'svg', and 'eps'. If no
      extension is provided, '.png' is used by default.
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
    os.makedirs(out_dir, exist_ok=True)
    if not file.endswith(('png', 'jpg', 'svg', 'eps')):
        file = file + '.png'
    fig.savefig(file, **kwargs)


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
        vmins = []
        vmaxs = []
        for ax in axs:
            if not ax.images:
                continue
            im = ax.images[-1]  # most recently added image
            try:
                arr = np.asarray(im.get_array())
                vmins.append(np.nanmin(arr))
                vmaxs.append(np.nanmax(arr))
            except Exception:
                v0, v1 = im.get_clim()
                vmins.append(v0)
                vmaxs.append(v1)
        if vmins:  # at least one image across all axes
            vmin = min(vmins)
            vmax = max(vmaxs)
            for ax in axs:
                if not ax.images:
                    continue
                ax.images[-1].set_clim(vmin, vmax)

    return


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

    # Check if the mask dimensions match the plotted image dimensions
    image_shape = ax.images[0].get_array().shape
    if mask.shape != image_shape:
        raise ValueError(f"Mask dimensions {mask.shape} do not match the plotted image dimensions {image_shape}.")

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
