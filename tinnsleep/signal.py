import numpy as np
import logging
from tinnsleep.validation import is_valid_ch_names


def rms(epochs, axis=-1):
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode over the samples.

    .. math::

        rms = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(x_{i})^{2}}

    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_electrodes, n_samples)
        the epochs for the estimation
    axis : None or int or tuple of ints, optional (default: -1)
        Axis or axes along which the means are computed.

    Returns
    -------
    RMS : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    """

    return np.sqrt(mean_power(epochs, axis=axis))


def mean_power(epochs, axis=-1):
    """Mean power for each epoch and each electrode
    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_electrodes, n_samples)
        the epochs for the estimation
    axis : None or int or tuple of ints, optional (default: -1)
        Axis or axes along which the means are computed.

    Returns
    -------
    mean_power : ndarray, shape (n_trials, n_electrodes)
        Mean Power
    """
    return np.mean(epochs ** 2, axis=axis)


def power_ratio(epochs, labels, axis=-1):
    """Mean power ratio for each electrode

    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_electrodes, n_samples)
        the epochs for the estimation
    axis : None or int or tuple of ints, optional (default: -1)
        Axis or axes along which the means are computed.
    labels : ndarray | list, shape (n_trials,)
        A list of boolean. True are for numerator, False for the denominator

    Returns
    -------
    power_ratio : ndarray, shape (n_electrodes,)
        Mean Power
    """
    nb_num = np.count_nonzero(labels)
    nb_epochs = len(labels)
    if (nb_num == 0) or (nb_num == nb_epochs):
        raise ValueError("labels should have at least one True and one False")

    pow = mean_power(epochs, axis=axis)
    return np.mean(pow[labels], axis=0)/np.mean(pow[np.invert(labels)], axis=0)


def is_good_epochs(epochs, **kwargs):
    """Test if epochs are good according to reject and flat by on intra-epoch min-max thresholding.

    Parameters
    ----------
    epochs: ndarray, shape (n_epochs, n_channels, n_samples)
        the epochs to test
    ch_names: array-like, shape (n_channels,)
        list of str corresponding of the name of each channel
    channel_type_idx: dict
        dictionary with channel category as key with the index of channels, e.g.
        >>> channel_type_idx = dict(eeg=[0, 1, 2, 4], eog=[5, 6])  # first 4 channels eeg and two last eog
    rejection_thresholds: dict | None
        the rejection threshold used for bad channel per channel_type, e.g.
        >>> rejection_thresholds = dict(eog=150e-6, eeg=150e-6, emg=50e-3)
        If None, bad channels won't be tested
    flat_thresholds: dict | None
        the flat threshold used for bad channel per channel_type, e.g.
        >>> flat_thresholds = dict(eog=5e-6, eeg=5e-6, emg=50e-6)
        If None, flat channels won't be tested
    full_report : bool (default: False)
        If full_report=True, it will give True/False as well as a list of all offending channels.

    Returns
    -------

    """

    kwargs["full_report"] = True
    labels = []
    bad_lists = []
    for epoc in epochs:
        [label, bad_list] = _is_good_epoch(epoc, **kwargs)
        labels.append(label)
        bad_lists.append(bad_list)

    return labels, bad_lists

def _is_good_epoch(data, ch_names=None,
                   channel_type_idx=None,
                   rejection_thresholds=None,
                   flat_thresholds=None,
                   full_report=False,
                   ignore_chs=[]):
    """Test if data segment data is good according to reject and flat based on min-max thresholding.

    see ``tinnsleep.signal.is_good_epochs`` for detailed documentation

    Inspired and extended from ``mne.Epochs._is_good()`` by Louis Korczowski, 2020

    TODO: pretty unefficient because of the loop for each epoch and each test (bad and flat).
    TODO: Everything could be done very fast on all epoch with ONE line actually (not urgent)
    """

    n_channels, n_samples = data.shape

    ch_names = is_valid_ch_names(ch_names, n_channels)

    bad_list = list()
    has_printed = False
    checkable = np.ones(len(ch_names), dtype=bool)
    checkable[np.array([c in ignore_chs
                        for c in ch_names], dtype=bool)] = False

    # check data for each type of threshold (here 'bad' and 'flat')
    for refl, f, t in zip([rejection_thresholds, flat_thresholds], [np.greater, np.less], ['bad', 'flat']):
        if refl is not None:  # performs test only if threshold exists
            for key, thresh in refl.items():  # performs test for each category of channel (e.g. 'eeg' or 'meg')
                idx = channel_type_idx[key]   # get indexes for each category
                name = key.upper()
                if len(idx) > 0:
                    e_idx = data[idx]
                    deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)   # compute min-max
                    checkable_idx = checkable[idx]                           # take only channel to check
                    idx_deltas = np.where(np.logical_and(f(deltas, thresh),  # find bad channels
                                                         checkable_idx))[0]
                    if len(idx_deltas) > 0:
                        ch_name = [ch_names[idx[i]] for i in idx_deltas]
                        if (not has_printed):
                            logging.info('    Rejecting %s epoch based on %s : '
                                        '%s' % (t, name, ch_name))
                            has_printed = True
                        # return channels names if bad
                        if not full_report:
                            return False
                        else:
                            bad_list.extend(ch_name)

    if not full_report:
        return True
    else:
        if bad_list == []:
            return True, None
        else:
            return False, bad_list
