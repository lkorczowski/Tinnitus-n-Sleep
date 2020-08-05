import mne
from tinnsleep.utils import epoch
import numpy as np
import os
import pandas as pd
from datetime import datetime
import logging
LOGGER = logging.getLogger(__name__)


def CreateRaw(data, sfreq, ch_names, montage=None, ch_types='misc'):
    """Generate a mne raw structure based on hardcoded info for bruxisme data

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
     sfreq: float
         sample rate (in Hz)
    ch_names : list of str | int
        Channel names. If an int, a list of channel names will be created
        from ``range(ch_names)``.
    montage: None | str | DigMontage
        A montage containing channel positions. If str or DigMontage is specified, the channel info will be updated
        with the channel positions. Default is None. See also the documentation of mne.channels.DigMontage for more
        information.
    ch_types : list of str | str
        Channel types, default is ``'misc'`` which is not a
        :term:`data channel <data channels>`.
        Currently supported fields are 'ecg', 'bio', 'stim', 'eog', 'misc',
        'seeg', 'ecog', 'mag', 'eeg', 'ref_meg', 'grad', 'emg', 'hbr' or 'hbo'.
        If str, then all channels are assumed to be of the same type.

    Returns
    -------
    raw: Instance of mne.Raw
        the signal
    """

    if montage is None:
        montage = 'standard_1020'
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage)
    return raw


def RawToEpochs_sliding(raw, duration, interval, picks=None):
    """Generate an epoch array from mne.Raw given the duration and interval (in samples) using sliding window.

    Parameters
    ----------
    raw: Instance of mne.Raw
        the signal
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    picks: str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted as channel indices. In lists, channel
        type strings (e.g., ['meg', 'eeg']) will pick channels of those types, channel name strings
        (e.g., ['MEG0111', 'MEG2623'] will pick the given channels. Can also be the string values “all” to pick all
        channels, or “data” to pick data channels. None (default) will pick good data channels Cannot be None if ax
        is supplied.If both picks and ax are None separate subplots will be created for each standard channel
        type (mag, grad, and eeg).

    Returns
    -------
    epochs: ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `raw`. Epochs are in the first dimension.
    """

    raw = raw.copy().pick(picks=picks)
    return epoch(raw.get_data(), duration, interval, axis=1)


def CleanAnnotations(raw):
    """Clean annotations from existing mne.Raw if exists
    
    Parameters
    ----------
    raw: Instance of mne.Raw
        the signal
        
    Returns
    -------
    raw: Instance of mne.Raw
        the signal without annotations
    """
    if len(raw.annotations) > 0:
        raw.annotations.delete(np.arange(0, len(raw.annotations)))
    return raw


def AnnotateRaw_sliding(raw, labels, dict_annotations={1: "bad EPOCH"}, duration=50, interval=50, merge=False):
    """Annotate mne.Raw data based on an labels with a sliding window strategy

    Parameters
    ----------
    raw: Instance of mne.Raw
        the signal
    labels: array-like, shape (n_annotations,)
        A array of labels code to annotate (e.g. ints or booleans)
    dict_annotations: dict (default: {1: "bad EPOCH"})
        Map the labels code to annotation description. By default, 1 are converted to "bad EPOCH".
        If None or if the key doesn't exist, the labels are added to the dictionary without a description.
    duration: int
        Number of elements (i.e. samples) for all annotations.
    interval: int
        Number of elements (i.e. samples) to move for the next annotations (if interval>=duration, no overlap).
    merge: bool (default: False)
        if True, will merge successive labels with same key together.


    Returns
    -------
    raw: Instance of mne.Raw
        the signal

    """
    # if the raw is too short
    total_length = interval * (len(labels) - 1) + duration
    if raw.__len__() < total_length:
        raise ValueError(f"Total length ({total_length}) exceed length of raw ({raw.__len__()})")

    # if the key doesn't exist, it just create dictionary with the description being the label
    for label in np.unique(labels):
        if not label == 0:
            if label not in dict_annotations.keys():
                dict_annotations[label] = str(label)

    if (duration < 1) | (interval < 1):
        raise ValueError("Invalid range for parameters")

    for k, label in enumerate(labels):
        if label in dict_annotations:
            if merge:
                if k == 0:
                    start_epoch = k
                elif label != labels[k-1]:
                    start_epoch = k

                if k == (len(labels)-1):
                    end_epoch = k
                    raw.annotations.append([interval * start_epoch / raw.info["sfreq"]],
                                           [(end_epoch - start_epoch + 1) * duration / raw.info["sfreq"]],
                                           dict_annotations[label])
                elif label != labels[k+1]:
                    end_epoch = k
                    raw.annotations.append([interval * start_epoch / raw.info["sfreq"]],
                                           [(end_epoch - start_epoch + 1) * duration / raw.info["sfreq"]],
                                           dict_annotations[label])
            else:
                start_epoch = k
                end_epoch = k
                raw.annotations.append([interval * start_epoch / raw.info["sfreq"]],
                                       [(end_epoch - start_epoch+1) * duration / raw.info["sfreq"]],
                                       dict_annotations[label])

    return raw


def convert_Annotations(annotations):
    """convert the instance mne.Annotations to a list of dict to make it iterable

    Parameters
    ----------
    annotations: dict
        e.g. mne.Annotations
        accepted keys: ['onset', 'duration', 'description', 'orig_time']
        each key can have a ndarray but the length

    Returns
    -------
    converted_annot: list of dict
        the converted annotations
    """
    converted_annot = []
    for annot in annotations:
        converted_annot.append(annot)

    return converted_annot


def align_labels_with_raw(labels, labels_timestamp, raw_info_start_time, raw_times=None, time_format='%H:%M:%S'):
    """Align timestamped labels DataFrame with the timestamps of mne.Raw.

    NOTE: timestamps are managed with second-precision.

    Align labels which has absolute datetime timestamps with relative reference of given raw. The new reference will be
    the start of the recording using ``raw_info_start_time`` parameter (see example below).
    Return some logging warning if the sampling of ``labels_timestamp`` is non-uniform or if ``labels_timestamp``
    doesn't match the recording length (too short or too long).

    Example
    -------
    >>> from tinnsleep.data import align_labels_with_raw
    >>> import pandas as pd
    >>> raw = mne.io.read_raw_edf(".data/raw.edf")
    >>> df_labels = pd.read_csv(sleep_file, sep=";")
    >>> labels_timestamp = align_labels_with_raw(df_labels["labels"], df_labels["date"], raw.info["meas_date"].time(), raw.times())

    Parameters
    ----------
    labels: ndarray
        array containing any type of label.
    labels_timestamp: ndarray
        array containing the timestamps of labels with the format '%H:%M:%S' (example : '23:25:20')
    raw_info_start_time: datetime.time instance
        can be generate by raw.info["meas_date"].time()
    raw_times: ndarray (optional, default: None)
        the vector of index of the raw instance.
        can be generate by raw.times()
        if given, the function will return a warning if the labels_timestamp are much shorter that the mne.Raw.
        Doesn't change the output.

    Returns
    -------
    labels_timestamp: ndarray
        an array of timestamps in seconds relative to the start of the mne.Raw instance.

    """
    if len(labels) != len(labels_timestamp):
        raise ValueError(f"labels ({len(labels)},) and labels_timestamp ({len(labels_timestamp)},) should be same length")

    delta_start = (datetime.strptime(str(labels_timestamp[0]), time_format) - \
                   datetime.strptime(str(raw_info_start_time), time_format)).total_seconds() \
                  % (3600 * 24)

    tmp = pd.to_datetime(pd.Series(labels_timestamp))
    labels_timestamp = ((tmp - tmp[0]).astype('timedelta64[s]') + delta_start).mod(3600 * 24).values

    # OPTIONAL CHECKS
    # the warnings shouldn't be deal-breaker in most of the situation but aweness of those might be important
    # TODO: MAYBE REMOVE ?
    interval = np.unique(np.diff(labels_timestamp))
    if len(interval) > 1:
        LOGGER.warning(f"non uniform interval (values: {interval}), taking median")
        interval = np.median(np.diff(labels_timestamp))
    else:
        interval = interval[0]

    if delta_start > interval:
        LOGGER.warning(f"delta_start {delta_start}")

    # optional check
    if raw_times is not None:
        delta_end = raw_times[-1] - (labels_timestamp[-1] + interval)
        if delta_start > interval:
            LOGGER.warning(f"delta_end ({delta_end}) > interval ({interval})")

    return labels_timestamp
