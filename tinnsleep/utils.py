import numpy as np
from numpy.lib import stride_tricks
from sklearn.utils.validation import check_array
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)

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

    return int(np.ceil((N-T+1) / I))


def merge_labels_list(list_valid_labels, n_epoch_final, merge_fun=np.all):
    """Merge a list of list of booleans labels into a unique array of booleans of size `n_epoch_final` by resampling
    each list by interpolation. The resampled list of booleans are merged using the logical `merge_fun`.

    ----------
    list_valid_labels : list of list of booleans
        lists obtained from different preprocessing loops.
    n_epoch_final : int
        length of the output ndarray desired
    merge_fun : function (default: numpy.all)
        a function for merging rows of a boolean matrix. Called with parameters axis=-1`

    Returns`
    -------
    valid_labels : ndarray of shape (n_epoch_final, )
        Merged list_valid_labels
    """
    nb_list = len(list_valid_labels)
    valid_labels = np.ones((n_epoch_final, nb_list)) * np.nan
    for i in range(nb_list):

        n_epoch_before = len(list_valid_labels[i])

        resampling_factor = float(n_epoch_before / n_epoch_final)
        if resampling_factor.is_integer() and (resampling_factor != 1):
                resampling_factor = int(resampling_factor)
                valid_labels[:, i] = [np.sum(list_valid_labels[i][j * resampling_factor:(j + 1) * resampling_factor])
                                      /
                                      resampling_factor == 1 for j in range(n_epoch_final)]

        else:
            logger.warning("Interpolating non proportional list, expecting to have non-uniform shift across recording")
            # Start Linear space
            x = np.linspace(0, n_epoch_before, num=n_epoch_before, endpoint=True)
            # Projection linear space
            xnew = np.linspace(0, n_epoch_before, num=n_epoch_final, endpoint=True)
            # Interpolation object definition
            f1 = interp1d(x, list_valid_labels[i], kind='nearest')
            valid_labels[:, i] = f1(xnew)

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
            labels.insert(i, False)   # inserting False (neutral) values in the labels array where they were deleted
    return labels

