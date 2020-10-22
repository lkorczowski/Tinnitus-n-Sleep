import numpy as np
from numpy.lib import stride_tricks
from sklearn.utils.validation import check_array
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)
import datetime


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
    `data` that has shape `(num_epochs, ..., size, ...)`. Dimensions other than
    the one represented by `axis` do not change.

    Parameters
    ----------
    data: array_like, shape (n_channels, n_samples)
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
    ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `data`. Epochs are in the first dimension.
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

    return int(np.ceil((N - T + 1) / I))


def merge_labels_list(list_valid_labels, n_epoch_final, merge_fun=np.all, kind='nearest'):
    """Merge a list of list of booleans labels into a unique array of booleans of size `n_epoch_final` by resampling
    each list by interpolation. The resampled list of booleans are merged using the logical `merge_fun`.

    ----------
    list_valid_labels : list of list of booleans
        lists obtained from different preprocessing loops.
    n_epoch_final : int
        length of the output ndarray desired
    merge_fun : function (default: numpy.all)
        a function for merging rows of a boolean matrix. Called with parameters axis=-1`
    kind : string (default: 'nearest')
        Only used when labels in the list are non-proportionnal
        Specifies the kind of interpolation as a string (see ``scipy.interpolate.interp1d`` documentation.)
        'previous' or 'nearest' should be best.

    Returns`
    -------
    valid_labels : ndarray of shape (n_epoch_final, )
        Merged list_valid_labels
    """
    nb_list = len(list_valid_labels)
    valid_labels = np.ones((n_epoch_final, nb_list)) * np.nan
    for i in range(nb_list):

        n_epoch_before = len(list_valid_labels[i])

        resampling_factor = float(n_epoch_final / n_epoch_before)

        # downsampling
        if (1 / resampling_factor).is_integer() and (resampling_factor != 1):
            resampling_factor = int(1 / resampling_factor)
            valid_labels[:, i] = [np.sum(list_valid_labels[i][j * resampling_factor:(j + 1) * resampling_factor])
                                  /
                                  resampling_factor == 1 for j in range(n_epoch_final)]
        # upsampling
        elif (resampling_factor).is_integer() and (resampling_factor != 1):
            resampling_factor = int(resampling_factor)
            for j in range(n_epoch_before):
                valid_labels[j * resampling_factor:(j + 1) * resampling_factor, i] = list_valid_labels[i][j]
        # nothing
        elif (resampling_factor == 1):
            valid_labels[:, i] = list_valid_labels[i]

        # interpolation
        else:
            logger.warning("Interpolating non proportional list, expecting to have non-uniform shift across recording")
            # Start Linear space
            x = np.linspace(0, n_epoch_before, num=n_epoch_before, endpoint=True)
            # Projection linear space
            xnew = np.linspace(0, n_epoch_before, num=n_epoch_final, endpoint=True)
            # Interpolation object definition
            valid_labels[:, i] = resample_labels(list_valid_labels[i], xnew, x=x, kind=kind)

    return merge_fun(valid_labels, axis=-1)


def fuse_with_classif_result(check_imp, labels):
    """Adds at the good indexes the missing elements of labels because of the use of check_RMS

    Parameters
    ----------
    check_imp : list of list , shape (nb_epochs, nb_electrodes)
         0s and 1s 0s marking bad channels for the designated epoch
    labels : ndarray, shape (n_trials - nb_bad_epochs,)
        array of the classification prediction results

    Returns
    -------
    mod_labels : ndarray, shape (n_trials,)
        Modified labels to fit a length of n_trials
    """
    labels = labels.tolist()
    for i in range(len(check_imp)):
        if np.mean(check_imp[i]) == 1:
            labels.insert(i, False)  # inserting False (neutral) values in the labels array where they were deleted
    return labels


def crop_to_proportional_length(epochs, valid_labels):
    """align number of epochs and a list of valid_labels to be proportional"""
    # compute all resampling factors
    resampling_factors = [int(len(epochs) / len(i)) for i in valid_labels]

    # find the common denominator
    min_labels = min([len(i) * j for (i, j) in zip(valid_labels, resampling_factors)])
    assert (len(epochs) - min_labels) <= max(
        resampling_factors), f"shift of {len(epochs) - min_labels} epochs max ({max(resampling_factors)}), please check that all duration are proportional"
    epochs = epochs[:min_labels]  # crop last epochs
    valid_labels_crop = [i[:int(min_labels / j)] for (i, j) in
                         zip(valid_labels, resampling_factors)]  # crop valid_labels
    assert len(epochs) == min_labels, f"something went wrong when cropping"
    valid_labels_crop = merge_labels_list(valid_labels_crop, len(epochs))
    return epochs, valid_labels_crop


def resample_labels(labels, xnew, x=None, kind='previous'):
    """Resample a list of labels using linear interpolation.
    To do so, ``labels`` are assigned to ``x`` timestamps and interpolated to new ``xnew`` timestamps.
    By default, user just need to give ``labels`` and a int as ``xnew`` for the length of the output labels.
    ``x`` and ``xnew`` can also be given as a old and new timestamp for interpolating complex value.
    By default, the new labels are associated with the previous observed label :

    Example
    -------
    >>> resample_labels(['A', 'B', 'C'], 2) # reduce labels
    array(['A', 'B'], dtype='<U1')

    >>> resample_labels(['A', 'B', 'C'], [0, 1]) # identical to previous example
    array(['A', 'B'], dtype='<U1')


    >>> xold = np.linspace(0, 2, 3)
    >>> xnew = np.linspace(0, 2, 2) # resample same interval including edges
    >>> resample_labels(['A', 'B', 'C'], xnew, x=xold)
    array(['A', 'C'], dtype='<U1')

    Parameters
    ----------
    labels : ndarray-like, shape (nb_labels, )
        list of labels (containing any type or mixed type)
    xnew : int | ndarray-like, shape (nb_labels_new,)
        If int : Number of elements in the new list of labels
        If ndarray-like : the timestamp of the new labels
    x : ndarray-like (default: range(len(labels)) )
        The timestamp of the old labels
    kind : string (default: 'previous')
        Specifies the kind of interpolation as a string (see ``scipy.interpolate.interp1d`` documentation.)
        'previous' or 'nearest' should be best.

    Returns
    -------
    labels_new : ndarray, shape (len(xnew),)
        Modified labels that fit to the new timestamps xnew

    """
    if isinstance(labels, list):
        labels = np.array(labels)
    n_labels = len(labels)
    if isinstance(xnew, int):
        xnew = np.linspace(0, n_labels, xnew, endpoint=False)

    if x is None:
        x = range(len(labels))
    elif len(x) != len(labels):
        raise ValueError(f"Number of labels is {len(labels)}, number of associated timestamps is {len(x)}")

    f = interp1d(x, range(len(labels)), kind=kind, fill_value=(0, len(x) - 1), bounds_error=False)

    return labels[f(xnew).astype(int)]


def label_report(labels):
    """Returns a dictionary with the count and the ratio of each unique labels from a list/array."""
    if labels is None:
        labels = []
    if isinstance(labels, list):
        labels = np.array(labels)

    report = dict()
    for label in np.unique(labels):
        report[f"{label} count"] = np.sum(labels == label)
        report[f"{label} ratio"] = np.sum(labels == label) / len(labels)
    return report


def merge_label_and_events(events_time, labels, time_interval):
    """Resample a list of labels at events_time from labels sampled at time_interval.

    Parameters
    ----------
    events_time : ndarray-like, shape (nb_resampled_labels, )
        list of timestamps in seconds.
    labels : ndarray-like, shape (nb_labels, )
        list of labels (containing any type or mixed type)
    time_interval : float
        the initial interval between two labels.

    Parameters
    ----------
    resampled_labels : ndarray-like, shape (nb_resampled_labels, )
        the resampled labels extracted from ``labels`` at time ``events_time``

    """
    x = np.linspace(0, (len(labels) - 1) * time_interval, len(labels))
    return resample_labels(labels, events_time, x)


def print_dict(data_dict):
    for key, value in data_dict.items():
        print(key, ' : ', value)


def round_time(dt=None, round_to=60):
    """
    Parameters
    ----------
    dt : datetime instance at second accuracy (doesn't work bellow)
        a datetime
    round_to : float
        number of second to round
        example : `round_to=60` rounds dt to the nearest minute

    Returns
    -------
    dt_rounded : datetime instance
        datetime rounded to the closest date using round_to interval.
    """
    if dt is None:
        dt = datetime.datetime.now()

    if round_to < 1:
        raise ValueError("round_time doesn't manage rounding under second")

    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)
