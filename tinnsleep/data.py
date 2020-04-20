import mne
from tinnsleep.utils import epoch
import numpy as np


def CreateRaw(data, ch_names, montage=None, ch_types=None):
    """Generate a mne raw structure based on hardcoded info for bruxisme data

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    ch_names : list of str | int
        Channel names. If an int, a list of channel names will be created
        from ``range(ch_names)``.
    montage: None | str | DigMontage
        A montage containing channel positions. If str or DigMontage is specified, the channel info will be updated
        with the channel positions. Default is None. See also the documentation of mne.channels.DigMontage for more
        information.
    ch_types: ‘mag’ | ‘grad’ | ‘planar1’ | ‘planar2’ | ‘eeg’ | None | list
        The channel type to plot. For ‘grad’, the gradiometers are collec- ted in pairs and the RMS for each pair
        is plotted. If None (default), it will return all channel types present. If a list of ch_types is provided,
        it will return multiple figures.

    Returns
    -------
    raw: Instance of mne.Raw
        the signal
    """
    if ch_types is None:
        ch_types = ['eeg']
    ch_types = ch_types * len(ch_names)
    sfreq = 200
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

    raw = raw.pick(picks=picks)
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


def AnnotateRaw_sliding(raw, labels, dict_annotations={1: "bad EPOCH"}, duration=50, interval=50):
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
            raw.annotations.append([interval * k / raw.info["sfreq"]],
                                   [duration / raw.info["sfreq"]],
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
