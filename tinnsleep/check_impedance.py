import mne
import numpy as np
from tinnsleep.utils import epoch


# Create Raw file
def Impedance_thresholding_sliding(data, duration, interval, THR=4000, axis=-1):
    """Tags each electrode of each epoch of a recording with a label, 1 meaning fine, 0 meaning Bad epoch for this
    channel
    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    THR : float
        Threshold for impedance checking. If mean value of an epoch above for a certain channel, this epoch will be
        flagged for this channel.
    axis : float
        Axis of the averaging, to adapt with the epochs table shape
    Returns
    -------
    check_imp list of list of booleans shape (nb_epochs, nb_electrodes)
        True marking bad channels for the designated epoch
    """
    # Epoching of the data
    epochs = epoch(data, duration, interval)
    # Averaging impedance signal per epoch per electrode
    mean_imp = np.mean(abs(epochs), axis=axis)
    # Thresholding
    check_imp = np.where(mean_imp > THR, True, False)
    return check_imp


def check_RMS(X, check_imp):
    """Modifies a RMS function output list by either suppressing the epoch as all channels are bad or modifying the
    values of the bad channels by the average value of the others.
    Parameters
    ----------
    X : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    check_imp list of list of booleans,
        shape (nb_epochs, nb_electrodes) True marking bad channels for the designated epoch

    Returns
    -------
    X : ndarray, shape (n_trials, n_electrodes)
    Modified Root Mean Square Amplitude
    """
    # Verify that X and check_imp have the same shape
    # TO DO Throw an error if else
    if not (len(X) == len(check_imp) and len(X[0]) == len(check_imp[0])) :
        raise ValueError("Inputs shapes don't match")

    mod_X=[]
    for i in range(len(X)):
        if not np.mean(check_imp[i]) == 1:  # if there is at least one good channel
            remaining_chan = []
            bad_chan_ind = []
            mod_X.append(X[i])
            for j in range(len(X[0])):    # get values of all good remaining channels
                if check_imp[i][j] == 0:
                    remaining_chan.append(X[i][j])
                else:
                    bad_chan_ind.append(j)
            rem_chan_av = np.mean(remaining_chan)   # get the average value of all good remaining channels
            for ind in bad_chan_ind:
                mod_X[-1][ind] = rem_chan_av   # replacing bad channels values with the mean of the remaining

    return mod_X

def fuse_with_classif_result(check_imp, labels):
    """Adds at the good indexes the missing elements of labels because of the use of check_RMS
    Parameters
    ----------
    check_imp list of list of booleans, shape (nb_epochs, nb_electrodes) True marking bad channels for the
    designated epoch
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


def create_annotation_sliding(check_imp, duration, interval, orig_time=0.0):
    """Create a list of annotations of all the bad epochs where all channels have abnormal impedance values to annotate
    the raw file
    Parameters
    ----------
    check_imp list of list of booleans, shape (nb_epochs, nb_electrodes)
        True marking bad channels for the designated epoch
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.

    Returns
    -------
    annotations : list of dictionaries
    Chronological list of bad epoch annotations
    """
    annotations = []
    for i in range(len(check_imp)):
        if np.mean(check_imp[i]) == 1:
            annotations.append({'onset': i * interval, 'duration': duration, 'description': "1", 'orig_time': orig_time})
    return annotations

def create_annotation_mne(check_imp):
    """Create a list of annotations of all the bad epochs where all channels have abnormal impedance values to annotate
    the raw file
    Parameters
    ----------
    check_imp list of list of booleans, shape (nb_epochs, nb_electrodes)
        True marking bad channels for the designated epoch
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.

    Returns
    -------
    annotations : list of dictionaries
    Chronological list of bad epoch annotations
    """
    return np.where(np.mean(check_imp, axis=-1) == 1, True, False)
