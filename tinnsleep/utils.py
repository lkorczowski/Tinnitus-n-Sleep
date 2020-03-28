import numpy as np
from numpy.lib import stride_tricks
from sklearn.utils.validation import check_array

def epoch(data, duration, interval, axis=-1):
    """ Small proof of concept of an epoching function using NumPy strides
    License: BSD-3-Clause
    Copyright: David Ojeda <david.ojeda@gmail.com>, 2018.
               Modified by Louis Korczowski <louis.korczowski@gmail.com>, 2020.
    Create a view of `a` as (possibly overlapping) epochs.
    The intended use-case for this function is to epoch an array representing
    a multi-channels signal with shape `(n_samples, n_channels)` in order
    to create several smaller views as arrays of size `(size, n_channels)`,
    without copying the input array.
    This function uses a new stride definition in order to produce a view of
    `a` that has shape `(num_epochs, ..., size, ...)`. Dimensions other than
    the one represented by `axis` do not change.

    Parameters
    ----------
    data: array_like
        Input array
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    axis: int
        Axis of the samples on `a`. For example, if `a` has a shape of
        `(num_observation, num_samples, num_channels)`, then use `axis=1`.
    Returns
    -------
    ndarray
        Epoched view of `a`. Epochs are in the first dimension.
    """
    data = np.asarray(data)
    data = check_array(data)
    if (duration < 1) | (interval < 1):
        raise ValueError("Invalid range for parameters")

    n_samples = data.shape[axis]
    n_epochs = (n_samples - duration) // interval + 1

    new_shape = list(data.shape)
    new_shape[axis] = duration
    new_shape = (n_epochs,) + tuple(new_shape)

    new_strides = (data.strides[axis] * interval,) + data.strides

    return stride_tricks.as_strided(data, new_shape, new_strides)

def compute_nb_epochs(N, T, I):
    """Return the exact number of expected windows based on the samples (N), window_length (T) and interval (I)
        Parameters
    ----------
    N: int
        total number of samples of the full signal to be epoched
    T: int
        duration of each epoch in samples
    I: int
        interval between the start of each epoch (if T == I, no overlap), always > 0

    Returns
    -------
    n_epochs: int
        estimated number of epochs
    """
    if (T < 1) | (I < 1):
        raise ValueError("Invalid range for parameters")

    return int(np.ceil((N-T+1) / I))
