import numpy as np
import logging

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

    return np.sqrt(np.mean(epochs ** 2, axis=axis))


def is_good_epoch(epochs, ch_names=None, channel_type_idx=None, reject=None, flat=None, full_report=False,
             ignore_chs=[], verbose=None):
    """Test if data segment e is good according to reject and flat.
    If full_report=True, it will give True/False as well as a list of all
    offending channels.
    Inspired and extended from ``mne.Epochs._is_good()`` by Louis Korczowski, 2020

    Parameters
    ----------
    epochs: ndarray, shape (n_epochs, n_channels, n_samples)
        the epochs to test
    ch_names:

    References
    ----------

    """

    ch_names = is_valid_ch_names(ch_names)

    bad_list = list()
    has_printed = False
    checkable = np.ones(len(ch_names), dtype=bool)
    checkable[np.array([c in ignore_chs
                        for c in ch_names], dtype=bool)] = False
    for refl, f, t in zip([reject, flat], [np.greater, np.less], ['', 'flat']):
        if refl is not None:
            for key, thresh in refl.items():
                idx = channel_type_idx[key]
                name = key.upper()
                if len(idx) > 0:
                    e_idx = epochs[idx]
                    deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                    checkable_idx = checkable[idx]
                    idx_deltas = np.where(np.logical_and(f(deltas, thresh),
                                                         checkable_idx))[0]

                    if len(idx_deltas) > 0:
                        ch_name = [ch_names[idx[i]] for i in idx_deltas]
                        if (not has_printed):
                            logging.info('    Rejecting %s epoch based on %s : '
                                        '%s' % (t, name, ch_name))
                            has_printed = True
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
