# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:18:57 2024

@author: Simon Kern (@skjerns)
"""
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sensors(
    values,
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
    layout = mne.channels.read_layout("Vectorview-all")
    positions = layout.pos[:, :2].T

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
