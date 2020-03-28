import mne
from tinnsleep.utils import epoch

#Create Raw file
def CreateRaw(data, ch_names, montage=None, ch_types=None):
    """Generate a mne raw structure based on hardcoded info for bruxisme data"""
    if ch_types is None:
        ch_types = ['eeg']
    ch_types = ch_types * len(ch_names)
    sfreq = 200
    if montage is None:
        montage = 'standard_1020'
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

def RawToEpochs_sliding(raw, duration, interval, picks=None):
    """Generate an Epoch structure from mne raw given the duration and interval by sliding window
    Parameters
    ----------
    a: array_like
        Input array
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
    epochs: ndarray
        Epoched view of `raw`. Epochs are in the first dimension.
    """

    raw = raw.pick(picks=picks)
    epochs = epoch(raw.get_data(), duration, interval, axis=1)
    return epochs
