import numpy as np
from tinnsleep.data import RawToEpochs_sliding
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.utils import fuse_with_classif_result, merge_labels_list
from tinnsleep.signal import rms
from tinnsleep.events.scoring import classif_to_burst, burst_to_episode, episodes_to_list
from tinnsleep.signal import is_good_epochs, power_ratio


def generate_bruxism_report(classif, time_interval, delim, min_burst_joining=3):
    """ Generates an automatic clinical bruxism report from a list of events

    Parameters
    ----------
    classif : list of booleans,
        output of a classification algorithm that detect non aggregated bursts from a recording
    time_interval: float,
        time interval in seconds between 2 elementary events
    delim: float,
        maximal time interval considered eligible between two bursts within a episode
    min_burst_joining : int
        minimal number of events to join to form an episode
    Returns
    -------
    report :  dict
    """
    report = {}
    recording_duration = len(classif) * time_interval
    report["Clean data duration"] = recording_duration
    report["Total burst duration"] = np.sum(classif) * time_interval
    li_burst = classif_to_burst(classif, time_interval)
    nb_burst = len(li_burst)
    report["Total number of burst"] = nb_burst
    report["Number of bursts per hour"] = nb_burst * 3600 / recording_duration
    li_episodes = burst_to_episode(li_burst, delim, min_burst_joining=min_burst_joining)
    nb_episodes = len(li_episodes)
    report["Total number of episodes"] = nb_episodes
    if nb_episodes > 0:
        report["Number of bursts per episode"] = nb_burst / nb_episodes
    else:
        report["Number of bursts per episode"] = 0
    report["Number of episodes per hour"] = nb_episodes * 3600 / recording_duration

    # Counting episodes according to types and listing their durations
    counts_type = [0, 0, 0]
    tonic = []
    phasic = []
    mixed = []
    for epi in li_episodes:
        if epi.is_tonic:
            counts_type[0] += 1
            tonic.append(epi.end - epi.beg)
        if epi.is_phasic:
            counts_type[1] += 1
            phasic.append(epi.end - epi.beg)
        if epi.is_mixed:
            counts_type[2] += 1
            mixed.append(epi.end - epi.beg)

    report["Number of tonic episodes per hour"] = counts_type[0] * 3600 / recording_duration
    report["Number of phasic episodes per hour"] = counts_type[1] * time_interval * 3600 / recording_duration
    report["Number of mixed episodes per hour"] = counts_type[2] * time_interval * 3600 / recording_duration

    # Mean durations of episodes per types, "nan" if no episode of a type recorded
    report["Mean duration of tonic episode"] = np.mean(tonic)
    report["Mean duration of phasic episode"] = np.mean(phasic)
    report["Mean duration of mixed episode"] = np.mean(mixed)
    return report


def generate_MEMA_report(classif, time_interval, delim):
    """ Generates an automatic clinical middle ear activition (MEMA) report from a list of events

    Parameters
    ----------
    classif : list of booleans,
        output of a classification algorithm that detect non aggregated bursts from a recording
    interval : float,
        time interval in seconds between 2 elementary events
    delim : float,
        maximal time interval considered eligible between two bursts within a episode
    Returns
    -------
    report : dict
    """
    report = {}
    recording_duration = len(classif) * time_interval
    report["Clean MEMA duration"] = recording_duration
    report["Total MEMA burst duration"] = np.sum(classif) * time_interval
    li_burst = classif_to_burst(classif, time_interval=time_interval)
    nb_burst = len(li_burst)
    report["Total number of MEMA burst"] = nb_burst
    report["Number of MEMA bursts per hour"] = nb_burst * 3600 / recording_duration
    li_episodes = burst_to_episode(li_burst, delim=delim, min_burst_joining=0)
    nb_episodes = len(li_episodes)
    report["Total number of MEMA episodes"] = nb_episodes
    if nb_episodes > 0:
        report["Number of MEMA bursts per episode"] = nb_burst / nb_episodes
    else:
        report["Number of MEMA bursts per episode"] = 0
    report["Number of MEMA episodes per hour"] = nb_episodes * 3600 / recording_duration

    episodes_duration = [(epi.end - epi.beg) for epi in li_episodes]

    report["Mean duration of MEMA episode"] = np.mean(episodes_duration)

    return report


def preprocess(raw, duration, interval,
                picks_chan="all",
                is_good_kwargs=None,
                filter_kwargs=None,
                Thresholding_kwargs=None,
                burst_to_episode_kwargs=None,
                merge_fun=None):
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
    burst_to_episode_kwargs : dict (default: None)
        parameters for ``tinnsleep.scoring.burst_to_episode`` class to merge Amplitude_Thresholding
    merge_fun : function (default: numpy.all(labels, axis=-1))
        function to merge is_good and Amplitude_Thresholding (True: valid epoch, False: artifact detected)
        by default, `numpy.all(labels, axis=-1)` is used: True only if all are valid_labels
        >>> merge_fun = lambda valid_labels: np.all(valid_labels, axis=-1)  # default

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `data`. Epochs are in the first dimension.
    valid_labels : lsit of booleans
        labels of the epochs as good (True) or bad (False) for future annotation and reporting
    log : dictionary
        logs of the preprocessing steps, including the number of epochs rejected at each step
    """
    raw = raw.copy().pick(picks=picks_chan).load_data()

    if isinstance(filter_kwargs, dict):
        raw = raw.filter(**filter_kwargs)
    elif filter_kwargs is None:
        pass  # do nothing
    else:
        raise ValueError('`filter_kwargs` a dict of parameters to pass to ``mne.raw.filter`` or None')

    if merge_fun is None:
        def merge_fun(foo):
            return np.all(foo, axis=-1)

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
        if isinstance(burst_to_episode_kwargs, dict) and np.any(RMSlabels):
            time_interval = interval/raw.info["sfreq"]
            RMSlabels = 0 < episodes_to_list(
                burst_to_episode(
                    classif_to_burst(RMSlabels, time_interval=time_interval),
                    **burst_to_episode_kwargs
                ), time_interval=time_interval, n_labels=X.shape[0]
            )
    else:
        RMSlabels = [False]*epochs.shape[0]
    suppressed_amp_thr = np.sum(RMSlabels)
    valid_labels = merge_fun(np.c_[np.invert(RMSlabels), amplitude_labels])
    suppressed_all = np.sum(np.invert(valid_labels))
    log = {"suppressed_is_good": suppressed_is_good,
           "suppressed_amp_thr": suppressed_amp_thr,
           "suppressed_overall": suppressed_all,
           "total_nb_epochs": len(valid_labels),
           "suppressed_ratio": suppressed_all/len(valid_labels)}

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
        report["Power Ratio"] = power_ratio(epochs[valid_labels], labels)

        labels = fuse_with_classif_result(np.invert(valid_labels),
                                          labels)  # add the missing labels removed with artefacts
        labs.append(labels)
        reps.append(report)

    parameters = dict()
    parameters['valid_labels'] = valid_labels
    parameters['THR_classif'] = THR_classif
    parameters['time_interval'] = time_interval
    parameters['delim'] = delim
    parameters['n_adaptive'] = n_adaptive

    return {"THR_classif": THR_classif, "labels": labs, "reports": reps, "log": log, "parameters": parameters}


def _cond_subclassif(ep_to_sub, labels_condition, time_interval):
    """subclassification of list of events into pure and combined events according to the labels_condition criterion
     ep_to_sub : list of episodes instances
            list of episodes to subclassify
    labels_condition : list of booleans
        list of events coming from episode and bursts from a chosen biosignal
    time_interval : float
        time interval between two labels of labels_condition
    Returns
    -------
    li_ep_c : list of booleans
        list of combined events grouped as episodes
    li_ep_p : list of booleans
        list of pure events grouped as episodes
    """
    # ---------------Subclassifying MEMA and bruxism events----------------------
    comb_ep = []
    pure_ep = []
    # merge episodes and compare with condition and artefacts
    for elm in ep_to_sub:
            if np.sum(labels_condition[int(elm.beg / time_interval):int(elm.end / time_interval)]) > 0:
                comb_ep.append(elm)
            else:
                pure_ep.append(elm)

    # ------------------
    # Pure episodes creation
    li_ep_p = episodes_to_list(pure_ep, time_interval, len(labels_condition))
    li_ep_p = np.where(li_ep_p != 0, True, False)
    # Combined episode creation
    li_ep_c = episodes_to_list(comb_ep, time_interval, len(labels_condition))
    li_ep_c = np.where(li_ep_c != 0, True, False)

    return li_ep_c, li_ep_p

def _labels_to_ep_and_bursts(labels, time_interval, delim_ep, min_burst_joining=0):
    """joining near bursts into episodes but keeping isolated bursts intact for future mutual conditioned analysis
        ----------
        labels: list of booleans
            list of events directly afer classification
        time_interval : float
            time interval between two labels
        delim_ep : float
            maximal tolerated time interval between 2 bursts to form an episode
        min_burst_joining_brux : int, (default : 0)
            minimal number of burst admitted to form an episode
        Returns`
        -------
        events : list of booleans
            list of events coming from episode or bursts from labels
        li_ep : list of episode instances
            list of episodes created from labels
        """
    # ------------grouping episodes and bursts together------------------------------------
    bursts = classif_to_burst(labels, time_interval=time_interval)
    li_ep = burst_to_episode(bursts, delim=delim_ep, min_burst_joining=min_burst_joining)
    events = episodes_to_list(li_ep, time_interval, len(labels))
    events = np.where(events != 0, True, False)
    return events, li_ep




def combine_brux_MEMA(labels_brux, time_interval_brux, delim_ep_brux, labels_MEMA,
                     time_interval_MEMA, delim_ep_MEMA,
                      min_burst_joining_brux=3, min_burst_joining_MEMA=0):
    """Combined analysis of bruxism and MEMA events to separate pure from combined events for both signals
        ----------
        labels_brux : list of booleans
            list of bruxism events directly afer classification
        time_interval_brux : float
            time interval between two labels of labels_brux
        delim_ep_brux : float
            maximal tolerated time interval between 2 bruxism bursts to form an episode
        labels_MEMA : list of booleans
            list of MEMA events directly afer classification
        time_interval_MEMA : float
            time interval between two labels of labels_MEMA
        delim_ep_MEMA : float
            maximal tolerated time interval between 2 MEMA bursts to form an episode
        min_burst_joining_brux : int, (default : 3)
            minimal number of burst admitted to form an episode for brux
        min_burst_joining_MEMA : int, (default :0)
            minimal number of burst admitted to form an episode for MEMA
        Returns`
        -------
        brux_comb_ep : list of booleans
            list of combined bruxism events grouped as episodes
        brux_pure_ep : list of booleans
            list of pure bruxism events grouped as episodes
        MEMA_comb_ep : list of booleans
            list of combined MEMA events grouped as episodes
        MEMA_pure_ep : list of booleans
            list of pure MEMA events grouped as episodes
        """
    # Putting labels inputs on the same sampling and epoching
    if len(labels_brux) != len(labels_MEMA):
        if len(labels_brux) < len(labels_MEMA):
            labels_brux = merge_labels_list ([labels_brux], len(labels_MEMA))
            time_interval = time_interval_MEMA
        else:
            labels_MEMA = merge_labels_list([labels_MEMA], len(labels_brux))
            time_interval = time_interval_brux
    else: #inputs of same length
        time_interval = time_interval_brux
    # Creating lists of episode and bursts for bruxism and MEMA
    brux_burst_ep, li_ep_brux = _labels_to_ep_and_bursts(labels_brux, time_interval, delim_ep_brux,
                                                         min_burst_joining=min_burst_joining_brux)
    MEMA_burst_ep, li_ep_MEMA = _labels_to_ep_and_bursts(labels_MEMA, time_interval, delim_ep_MEMA,
                                                         min_burst_joining=min_burst_joining_MEMA)

    # Conditionnal labelling of events
    MEMA_comb_ep, MEMA_pure_ep= _cond_subclassif(li_ep_MEMA,
                                                                 brux_burst_ep, time_interval)
    brux_comb_ep, brux_pure_ep = _cond_subclassif(li_ep_brux,
                                                                   MEMA_burst_ep, time_interval)

    return brux_comb_ep, brux_pure_ep, MEMA_comb_ep, MEMA_pure_ep
