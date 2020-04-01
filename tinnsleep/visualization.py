import matplotlib.pyplot as plt
import numpy as np


def plotTimeSeries(data,
                   ch_names=None,
                   sfreq=1,
                   scalings=None,
                   annotations=None,
                   ax=None,
                   **kwargs):
    """Advanced plotting of multidimensional time series from ndarray

    Parameters
    ----------
    data: array-line, shape (n_samples, n_dimension)
        multidimensional time series
    ch_names: list | iterable, shape (n_dimension,) | None
        the labels for the time series
    sfreq: float (default: 1)
        sample rate (in Hz)
    scalings: float | iterable, shape (n_dimension,)
        value between two channels
    annotations: a instance mne.Annotations | list of dictionary (default: {})
        a list of annotation or dictionary containing the following fields:
        {'onset': float (seconds), 'duration': float (seconds), 'description': str, orig_time': float (seconds)}
        Example:
        >>> # a list of one annotation starting after 0.5 second of duration 1.0 second named 'blink'
        >>> annotations = [{'onset': 0.5, 'duration': 1.0, 'description': "blink", 'origin_time': 0.0}]
    ax: a instance of ``matplotlib.pyplot.Axes`` (default: None)
        the axe where to save the fig. By default a new figure is generated.

    Returns
    -------
    n_epochs: int
        estimated number of epochs
    """
    shapeD = data.shape
    if len(shapeD) == 1:
        n_channels = 1
        n_samples = shapeD[0]
        data = np.expand_dims(data, axis=1)
    elif len(shapeD) == 2:
        n_channels = shapeD[1]
        n_samples = shapeD[0]
    elif len(shapeD) > 2:
        raise ValueError("data should be two-dimensional")

    if ch_names is None:
        ch_names = np.arange(0, n_channels)
    elif isinstance(ch_names, str):
        ch_names = [ch_names] * n_channels
    elif isinstance(ch_names, (np.ndarray, list)):
        if not len(ch_names) == n_channels:
            raise ValueError('ch_names should be same length as the number of channels of data')

    if ax is None:
        ax = plt.gca()
        fig = ax.figure
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        msg = "`ax` must be a matplotlib Axes instance or None"
        raise ValueError(msg)

    # remove median
    data = data - np.median(data, axis=0)

    if scalings is None:
        # distance between two lines: maximum of the 95% percentile of each channel
        scalings = np.max(np.quantile(np.abs(data), 0.975, axis=0))

    # calculate multidimensional shifts based on scalings
    shifts = np.linspace(0, 2 * scalings * (n_channels-1), n_channels)

    # align timeseries with new offsets
    data = data - shifts

    times = np.linspace(0, (n_samples-1) / sfreq, num=n_samples)

    # compute shift based on scalings

    ax.plot(times, data, label=ch_names, **kwargs)
    plt.yticks(-shifts, ch_names)
    plt.xlim(np.min(times), np.max(times))
    return fig, ax
