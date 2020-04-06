import numpy as np
import numpy.testing as npt

def is_valid_ch_names(ch_names, n_channels):
    """Check if ch_names is correct or generate a numerical ch_names list"""
    if (ch_names is None) or (ch_names == []):
        ch_names = np.arange(0, n_channels)
    elif isinstance(ch_names, str):
        ch_names = [ch_names] * n_channels
    elif isinstance(ch_names, (np.ndarray, list)):
        if not len(ch_names) == n_channels:
            raise ValueError('`ch_names` should be same length as the number of channels of data')
    else:
        msg = "`ch_names` must be a list or an iterable of shape (n_channels,) or None"
        raise ValueError(msg)
    return ch_names


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