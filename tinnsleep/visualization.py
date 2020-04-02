import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt


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
    else:
        msg = "`ch_names` must be a list or an iterable of shape (n_dimension,) or None"
        raise ValueError(msg)

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


def assert_ax_equals_data(data, ax, sfreq=1):
    """Return assert error if the ax is not comming from data

     Parameters
     ----------
     data: array-line, shape (n_samples, n_dimension)
         multidimensional time series
     ax: a instance of ``matplotlib.pyplot.Axes``
        the ax where the data were plotted
     sfreq: float (default: 1)
         sample rate (in Hz)
     """
    # check if correct values
    for n, line in enumerate(ax.get_lines()):
        x, y = line.get_data()
        # check if data correlated perfectly (there aren't equal due to transformation)
        npt.assert_approx_equal(np.corrcoef(y, data[:, n])[0][1], 1)  # data y-axis correlate to 1
        npt.assert_equal(x, np.linspace(0, (data.shape[0]-1)/sfreq, data.shape[0]))  # time x-axis match


def assert_x_labels_correct(data, expected_labels):
    """Return assert error if the ax is not comming from data

     Parameters
     ----------
     data: array-line, shape (n_samples, n_dimension)
         multidimensional time series
     expected_labels: a list of str
         the expected label names
     """
    # prepare data to double check
    data = data - np.median(data, axis=0)
    scalings = np.max(np.quantile(np.abs(data), 0.975, axis=0))

    # calculate multidimensional shifts based on scalings
    shifts = - np.linspace(0, 2 * scalings * (data.shape[1]-1), data.shape[1])

    # check if label position and values are correct
    locs, labels = plt.yticks()
    npt.assert_equal(locs, shifts)
    for k, label in enumerate(labels):
        assert label.get_text() == expected_labels[k], "labels are not the same"
