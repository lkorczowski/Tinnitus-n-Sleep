import numpy as np
import mne
from tinnsleep.config import Config
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.utils import fuse_with_classif_result
from tinnsleep.signal import rms
from tinnsleep.scoring import generate_bruxism_report, classif_to_burst, burst_to_episode, create_list_events
from tinnsleep.signal import is_good_epochs


def preprocess(raw, duration, interval,
                picks_chan="all",
                is_good_kwargs=None,
                filter_kwargs=None,
                Thresholding_kwargs=None,
                episode_kwargs=None,
                merge_fun=np.all):
    """Preprocesses raw and apply a list of operations for events detection. Each step can be muted by giving None.


    raw -> picks_chan -> filter -> epoch --------> is_good ----------------------------> labels -+
                                    |                                                            +--> merge labels-+
                                    +--> Amplitude_Thresholding -> labels -> merge into episodes-+                 |
                                    |                                                                              |
                                   OUT                                                                            OUT
    Parameters
    ----------
    raw: Instance of mne.Raw
        the signal to analyze
    duration : int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    picks_chan: str | list | slice | None (default: "all")
        Channels to include. Slices and lists of integers will be interpreted as channel indices. In lists, channel
        type strings (e.g., ['meg', 'eeg']) will pick channels of those types, channel name strings
        (e.g., ['MEG0111', 'MEG2623'] will pick the given channels. Can also be the string values “all” to pick all
        channels, or “data” to pick data channels. None (default) will pick all data channels.
    filter_kwargs : dict (default: None)
        parameters for ``mne.raw.filter()`` to apply to subset picks_chan
    is_good_kwargs : dict (default: None)
        parameters for ``tinnsleep.signal.is_good_epochs()`` to apply to subset picks_chan
    Thresholding_kwargs : dict (default: None)
        parameters for ``tinnsleep.classification.AmplitudeThresholding`` to apply
    episode_kwargs : dict (default: None)
        parameters for ``tinnsleep.episode`` class to merge
    merge_fun : function (default: numpy.all)
        function to merge is_good and Amplitude_Thresholding (True: valid epoch, False: artifact detected)

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `data`. Epochs are in the first dimension.
    valid_labels : lsit of booleans
        labels of the epochs as good (True) or bad (False) for future annotation and reporting
    log : dictionary
        logs of the preprocessing steps, including the number of epochs rejected at each step
    """

    if isinstance(filter_kwargs, dict):
        raw = raw.copy().filter(**filter_kwargs)
    elif filter_kwargs is None:
        pass  # do nothing
    else:
        raise ValueError('`filter_kwargs` a dict of parameters to pass to ``mne.raw.filter`` or None')

    if episode_kwargs is not None:
        raise ValueError(f"`episode_kwargs` algorithm not implemented yet")

    epochs = RawToEpochs_sliding(raw, duration, interval, picks=picks_chan)

    # Epoch rejection based on |min-max| thresholding
    if is_good_kwargs is not None:
        amplitude_labels, bad_lists = is_good_epochs(epochs, **is_good_kwargs)
    else:
        amplitude_labels, bad_lists = [True]*epochs.shape[0], []

    suppressed_is_good = np.sum(np.invert(amplitude_labels))

    # Epoch rejection based on Root Mean Square Amplitude thresholding
    if Thresholding_kwargs is not None:
        X = rms(epochs)
        pipeline = AmplitudeThresholding(**Thresholding_kwargs)
        RMSlabels = pipeline.fit_predict(X)
        #RMSlabels = # TODO: add episode fun
    else:
        RMSlabels = [False]*epochs.shape[0]
    suppressed_amp_thr = np.sum(RMSlabels)
    valid_labels = merge_fun(np.c_[np.invert(RMSlabels), amplitude_labels], axis=-1)
    suppressed_all = np.sum(np.invert(valid_labels))
    log = {"suppressed_is_good": suppressed_is_good, "suppressed_amp_thr": suppressed_amp_thr,
                                     "suppressed_overall": suppressed_all, "total_nb_epochs": len(valid_labels)}
    return epochs, valid_labels, log


def reporting(epochs, valid_labels, THR_classif, time_interval, delim, n_adaptive=0, log={}, generate_report=generate_bruxism_report):
    """creates clinical reports of bruxism out of a epoch array, for different thresholding values
    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `data`. Epochs are in the first dimension.
    valid_labels : list of booleans
        labels of the epochs as good (True) or bad (False) for future annotation and reporting
    THR_classif : list of a list floats
       list of couples of absolute and relative thresholds values of the classifier to test
       example: THR_classif=[[0,2],[0,3]]
    time_interval: float
        time interval in seconds between 2 elementary events
    delim: float, (default 3)
        maximal time interval considered eligible between two bursts within a episode
    n_adaptative : int (default: 0)
        number of epochs for adaptive baseline calculation
        if positive uses casual adaptive scheme
        if negative uses acasual forward-backward adaptive scheme
    log : dictionary (default: {})
        logs of the preprocessing steps, including the number of epochs rejected at each step
    generate_report: function (default: tinnsleep.scoring.generate_bruxism_report)
        function to convert labels into a report


    Returns
    -------
    dictionary
       Dictionary containing fields:
       - THR_classif: threshold(s) of classification tested
       - labels : list of labels of bursts for the epochs of the recording for each THR_classif in the same order
       - reports: list of clinical reports of the recording for each THR_classif tested in the same order
       - log : log of the pre-processing operations
    """
    labs = []
    reps = []
    # for each value of THR_classif, create a report and a list of labels
    for THR in THR_classif:
        X = rms(epochs[valid_labels])  # take only valid labels
        # -----------------Classification Forward ------------------------------------------------
        pipeline = AmplitudeThresholding(abs_threshold=THR[0], rel_threshold=THR[1], n_adaptive=abs(n_adaptive))
        labels = pipeline.fit_predict(X)
        if n_adaptive < -1:
            # -----------------Classification backward ---------------------------------------
            # Reversing epochs array, computing backward and reversing labels
            pipeline = AmplitudeThresholding(abs_threshold=THR[0], rel_threshold=THR[1], n_adaptive=abs(n_adaptive))
            labels_b = pipeline.fit_predict(X[::-1])[::-1]
            #-----------------foward-backward merge ---------------------------------------
            # Logical OR -- merged backward and forward
            labels = np.any(np.c_[labels, labels_b], axis=-1)

        report = generate_report(labels, time_interval, delim)
        labels = fuse_with_classif_result(np.invert(valid_labels),
                                          labels)  # add the missing labels removed with artefacts
        labs.append(labels)
        reps.append(report)

    return {"THR_classif": THR_classif, "labels": labs, "reports": reps, "log": log}



