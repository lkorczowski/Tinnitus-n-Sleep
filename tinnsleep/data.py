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


def align_labels_with_raw(labels_timestamp, raw_info_start_time=None, raw_times=None, time_format='%H:%M:%S'):
    """Convert timestamps to delta time and align timestamped labels DataFrame with the timestamps of mne.Raw.

    NOTE: timestamps are managed with second-precision by default, but automatically switch to
          `time_format='%H:%M:%S.%f'` for sub-second precision. Other format should be given using ``time_format``
          parameter.

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
    >>> labels_timestamp = align_labels_with_raw(df_labels["date"], raw.info["meas_date"].time(), raw.times)

    Parameters
    ----------
    labels_timestamp: ndarray
        array containing the timestamps of labels with the format by default '%H:%M:%S' (example : '23:25:20')
    raw_info_start_time: datetime.time instance
        can be generate by raw.info["meas_date"].time()
        WARNING: manage only format ``datetime.time`` with hour-minute-second (without micro-second).
    raw_times: ndarray (optional, default: None)
        the vector of index of the raw instance.
        can be generate by raw.times
        if given, the function will return a warning if the labels_timestamp are much shorter that the mne.Raw.
        Doesn't change the output.
    time_format: string (default: '%H:%M:%S')
        time format for reading ``labels_timestamp``.

    Returns
    -------
    labels_timestamp_delta: ndarray
        an array of timestamps in seconds relative to the start of the mne.Raw instance.

    """
    try:  # find first timestamp index
        start_idx = labels_timestamp.first_valid_index()  # for DataFrame
    except AttributeError:
        start_idx = 0  # for numpy

    try:  # read first timestamp
        start_labels = datetime.strptime(str(labels_timestamp[start_idx]), time_format)
    except ValueError:
        start_labels = datetime.strptime(str(labels_timestamp[start_idx]), time_format + ".%f")

    if raw_info_start_time is None:  # read recording date
        start_recording = start_labels
    else:
        start_recording = datetime.strptime(str(raw_info_start_time), "%H:%M:%S")

    delta_start = (start_labels - start_recording).total_seconds() \
                  % (3600 * 24)

    tmp = pd.to_datetime(pd.Series(labels_timestamp))
    labels_timestamp_delta = ((tmp - tmp[start_idx]).astype('timedelta64[s]') + delta_start).mod(3600 * 24).values

    # OPTIONAL CHECKS
    # the warnings shouldn't be deal-breaker in most of the situation but aweness of those might be important
    # TODO: MAYBE REMOVE ?
    interval = np.unique(np.diff(labels_timestamp_delta))
    if len(interval) > 1:
        interval_count = []
        for inter in interval:
            interval_count.append((inter, np.sum(np.diff(labels_timestamp_delta)==inter)))
        interval = np.median(np.diff(labels_timestamp_delta))
        LOGGER.info(f"non uniform interval (count: {interval_count}), taking median: {interval}")
        LOGGER.info(f"start time, file: {str(raw_info_start_time)} labels: {labels_timestamp[start_idx]}")
    else:
        interval = interval[0]

    if delta_start > interval:
        LOGGER.info(f"delta_start {delta_start}")

    # optional check
    if raw_times is not None:
        delta_end = raw_times[-1] - (labels_timestamp_delta[-1] + interval)
        if delta_start > interval:
            LOGGER.warning(f"delta_end ({delta_end}) > interval ({interval})")

    return labels_timestamp_delta


def read_sleep_file(sleep_file,
                      map_columns=None,
                      sep=";",
                      encoding="ISO-8859-1",
                      time_format='%H:%M:%S',
                      raw_info_start_time=None,
                      raw_times=None):
    """Read sleep .csv file and return labels and timestamps in seconds

    Parameters
    ----------
    sleep_file: str
        .csv file full path
    map_columns: dict
        convert columns to normalized columns name "Start Time", and "Sleep"
    sep: str (default: ";")
        separator of the csv file.
    encoding: str (default: "ISO-8859-1")
        encoding of the csv file.
    raw_info_start_time: datetime.time instance
        can be generate by raw.info["meas_date"].time()
        WARNING: manage only format ``datetime.time`` with hour-minute-second (without micro-second).
    raw_times: ndarray (optional, default: None)
        the vector of index of the raw instance.
        can be generate by raw.times
        if given, the function will return a warning if the labels_timestamp are much shorter that the mne.Raw.
        Doesn't change the output.
    time_format: string (default: '%H:%M:%S')
        time format for reading ``labels_timestamp``.


    Returns
    -------
    sleep_labels: ndarray

    sleep_label_timestamp: ndarray
    """
    if map_columns is None:
        map_columns = {"Horodatage": "Start Time",
                       "Sommeil": "Sleep",
                       "event": "Sleep",
                       "Event": "Sleep",
                       "begin": "Start Time"}

    df_labels = pd.read_csv(sleep_file, sep=sep, encoding=encoding)
    df_labels = df_labels.rename(columns=map_columns)

    # take only valid labels
    df_labels = df_labels[df_labels["Sleep"].str.contains("")==True]
    df_labels = df_labels[df_labels["Sleep"].str.contains("\[\]") == False]
    sleep_labels = df_labels["Sleep"].values

    # replace specific values
    sleep_labels[sleep_labels == '\x83veil'] = 'Wake'
    sleep_labels[sleep_labels == 'AWA'] = 'Wake'
    sleep_labels[sleep_labels.astype(str) == 'nan'] = 'Wake'


    sleep_label_timestamp = align_labels_with_raw(
        df_labels["Start Time"],
        raw_info_start_time=raw_info_start_time,
        raw_times=raw_times,
        time_format=time_format)

    return sleep_labels, sleep_label_timestamp

def read_etiology_file(etiology_file,
                      map_columns=None):
    """Read etiology .xlsx file and return panda dataframe with a column of possible etiology

    Parameters
    ----------
    etiology_file: str
        .xslx file full path
    map_columns: dict (default: None)
        dictionary to convert columns if needed, e.g. map_columns={"oldcolumn": "newcolumn"}


    Returns
    -------
    df_etiology: DataFrame
        rows: subject's file
        columns:
    sleep_label_timestamp: ndarray
    """
    df_raw = pd.read_excel(etiology_file)
    df_etiology = pd.DataFrame()

    # build subject list (should match exiting subject list in data_info.csv)
    df_etiology["subject"] = df_raw["Identifiant patient"]

#   1)	Oreille bouchée
#   Colonne CD de l'excel, champ "avezvousunsentimentdoreil", avec une réponse "oui" (tout le temps, fréquemment, parfois). Attention, pour certains patients (1DA15, 1DL12, 1MA16, 1SA14) on n'a pas cette info!
#   1SL21 et 1UC22 concernés, pas forcément 1HB20.
    mapping = {"Non, jamais ou presque": 0, "Oui, parfois": 1, "Oui, fréquemment": 2, "Oui, tout le temps": 3}
    key = "avezvousunsentimentdoreil"
    new_key = "obstructed_ear"
    df_etiology[new_key] = df_raw[key].replace(mapping) >= 1

    """2)	Otalgie
    Colonne CI de l'excel, champ "avezvousdesdouleursdansou", avec une réponse "oui" (tout le temps, fréquemment, parfois). 
    Attention, manque 1SA14
    1UC22, 1SL21, 1HB20 non concernés"""
    mapping = {"Non, jamais ou presque": 0, "Oui, parfois": 1, "Oui, fréquemment": 2, "Oui, tout le temps": 3}
    key = "avezvousdesdouleursdansou"
    new_key = "otalgy"
    df_etiology[new_key] = df_raw[key].replace(mapping) >= 1

    """3)	hyperacusis
    Colonne BS ou colonne BU = "Oui", (champs "estcequelexpositiondesson" ou "estcequedenombreuxbruitsd")
    1UC22 concerné, 1SL21 un petit peu concerné, on ne sait pas exactement pour 1HB20"""
    mapping = {"Oui": True, "Non": False}
    key1 = "estcequelexpositiondesson"
    key2 = "estcequedenombreuxbruitsd"
    new_key = "hyperacusis"
    df_etiology[new_key] = df_raw[key1].replace(mapping) | df_raw[key2].replace(mapping)

    """
    4)	Craquement mâchoire 3-4 fois par semaine 
    Colonne BZ de l'excel, champ "estcequelarticulationdevo", avec une réponse "oui" (tout le temps, fréquemment, parfois).
    Patient 1SL21 concerné, pas évident pour les 2 autres
    Attention, manque 1SA14 
    """
    mapping = {"Non, jamais ou presque": 0, "Oui, parfois": 1, "Oui, fréquemment": 2, "Oui, tout le temps": 3}
    key = "estcequelarticulationdevo"
    new_key = "jaw_popping"
    df_etiology[new_key] = df_raw[key].replace(mapping) >= 1

    """
    5)	Douleur ou fatigue muscle de la mastication ou de la face
    Colonne BX ou BY de l'excel, champ "ressentezvousdesdouleursa" ou "ressentezvousdelafatiguea", avec une réponse "oui" (tout le temps, fréquemment, parfois). 
    Patient 1SL21 concerné, pas évident pour les 2 autres
    Attention, manque 1SA14
    """
    mapping = {"Ne sait pas":0,"Non, jamais ou presque": 0, "Oui, parfois": 1, "Oui, fréquemment": 2, "Oui, tout le temps": 3}
    key1 = "ressentezvousdesdouleursa"
    key2 = "ressentezvousdelafatiguea"
    new_key = "jaw_pain_and_fatigue"
    df_etiology[new_key] = (df_raw[key1].replace(mapping) + df_raw[key2].replace(mapping))/2 >=1

    """
    6)	Modulation somato-sensorielle si >3
    Somme des valeurs colonnes DF à DP, test si résultat supérieur à 3. ATTENTION : colonne DQ correspond à "NON" du coup ne pas la compter. 
    Attention, manque 1SA14
    Oui pour 1UC22, faux pour les 2 autres"""
    mapping = {np.NaN: 0}
    key = ["vosacouphnesvarientilenin_0",
           "vosacouphnesvarientilenin_1",
           "vosacouphnesvarientilenin_2",
           "vosacouphnesvarientilenin_3",
           "vosacouphnesvarientilenin_4",
           "vosacouphnesvarientilenin_5",
           "vosacouphnesvarientilenin_6",
           "vosacouphnesvarientilenin_7",
           "vosacouphnesvarientilenin_8",
           "vosacouphnesvarientilenin_9",
           "vosacouphnesvarientilenin_10"]
    new_key = "somatosensory_modulation"
    df_etiology[new_key] = df_raw[key].replace(mapping).sum(axis=1) > 3

    """7)	Augmentation des acous suite aux siestes
    Colonne BN, champ "ressentezvousunehaussedev", réponse : "Oui je ressens une hausse franche de mon acouphène suite aux siestes" UNIQUEMENT 
    Oui pour 1HB20, 1SL21 et 1UC22"""

    mapping = {"Oui je ressens une hausse franche de mon acouphène suite aux siestes": 1}
    key = "ressentezvousunehaussedev"
    new_key = "nap_modulation"
    df_etiology[new_key] = df_raw[key].replace(mapping) == 1

    """
    8)	Ronflements (ou apnées sans ronflements)
    Colonne EI, champ "estcequevousronflez", réponse : "Oui"
    Attention, pour certains patients (1DA15, 1DL12, 1MA16, 1SA14) on n'a pas cette info!
    Oui pour 1HB20, 1SL21 et 1UC22"""

    mapping = {"Oui": 1, "Non": 0}
    key = "estcequevousronflez"
    new_key = "snoring"
    df_etiology[new_key] = df_raw[key].replace(mapping) == 1

    return df_etiology


