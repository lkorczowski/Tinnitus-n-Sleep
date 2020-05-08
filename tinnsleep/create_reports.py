import numpy as np
import mne
from tinnsleep.config import Config
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.check_impedance import create_annotation_mne, Impedance_thresholding_sliding, check_RMS, \
    fuse_with_classif_result
from tinnsleep.signal import rms
from tinnsleep.signal import is_good_epochs
from scipy.interpolate import interp1d
from tinnsleep.scoring import generate_bruxism_report, create_list_events, classif_to_burst, burst_to_episode


def preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=6000, get_log=False, filter="default"):
    """Preprocesses raw for reporting
    Parameters
    ----------
    raw: Instance of mne.Raw
        the signal to analyze
    picks_chan: array-like, shape (n_channels,)
        list of str corresponding of the name of each channel to analyze
    picks_imp:
        list of str corresponding of the name of each channel of impedance values associated with the channels to
        analyze
    duration : int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    THR_imp: float
        Threshold value for the impedance rejection algorithm
    get_log : OPTIONAL boolean default False
        if True, create a report of the preprocessing rejecting steps
    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, duration)
        Epoched view of `data`. Epochs are in the first dimension.
    valid_labels : lsit of booleans
        labels of the epochs as good (True) or bad (False) for future annotation and reporting
    log : dictionary
        logs of the preprocessing steps, including the number of epochs rejected at each step
    """

    # Epoch rejection based on impedance
    check_imp = Impedance_thresholding_sliding(raw[picks_imp][0], duration, interval, THR=THR_imp)
    impedance_labels = np.any(check_imp, axis=-1)
    suppressed_imp = np.sum(impedance_labels)

    raw = CreateRaw(raw[picks_chan][0], raw.info["sfreq"], picks_chan, ch_types='emg')  # pick channels and load

    # Filtering data
    if filter == "default":
        raw = raw.filter(l_freq = 20., h_freq = 99., n_jobs=4,
                         fir_design='firwin', filter_length='auto', phase='zero-double',
                         picks=picks_chan)
    elif isinstance(filter, dict):
        raw = raw.filter(**filter)
    elif filter is None:
        pass  # do nothing
    else:
        raise ValueError('`filter` should be default, a dict of parameters to pass to raw.filter, or None')

    # Creating epochs
    epochs = RawToEpochs_sliding(raw, duration=duration, interval=interval)

    # Epoch rejection based on |min-max| thresholding
    amplitude_labels, bad_lists = is_good_epochs(epochs, **params)

    suppressed_amp = np.sum(np.invert(amplitude_labels))

    # Reuniting the rejection algorithms
    valid_labels = np.all(np.c_[np.invert(impedance_labels), amplitude_labels], axis=-1)
    suppressed_all = np.sum(np.invert(valid_labels))

    if get_log:
        # Creating log report
        log = {"suppressed_imp_THR": suppressed_imp, "suppressed_amp_THR": suppressed_amp,
                                         "suppressed_overall": suppressed_all, "total_nb_epochs": len(valid_labels)}

        return epochs, valid_labels, log
    else:
        return epochs, valid_labels


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
            # Start Linear space
            x = np.linspace(0, n_epoch_before, num=n_epoch_before, endpoint=True)
            # Projection linear space
            xnew = np.linspace(0, n_epoch_before, num=n_epoch_final, endpoint=True)
            # Interpolation object definition
            f1 = interp1d(x, list_valid_labels[i], kind='nearest')
            valid_labels[:, i] = f1(xnew)

    return merge_fun(valid_labels, axis=-1)

def _cond_subclassif(ep_to_sub, labels_artifacts, labels_condition, time_interval):
    # ---------------Subclassifying MEMA and bruxism events----------------------
    comb_ep = []
    pure_ep = []
    compt_arti = 0
    # merge episodes and compare with condition and artefacts
    for elm in ep_to_sub:
        if np.sum(labels_artifacts[int(elm.beg / time_interval):int(elm.end / time_interval)]) == \
                (int(elm.end / time_interval) - int(elm.beg / time_interval)):
            if np.sum(labels_condition[int(elm.beg / time_interval):int(elm.end / time_interval)]) > 0:
                comb_ep.append(elm)
            else:
                pure_ep.append(elm)
        else:
            compt_arti += 1
    # ------------------
    # Pure episodes creation
    li_ep_p = create_list_events(pure_ep, time_interval, time_interval * len(labels_condition), boolean_output=True)
    # Combined episode creation
    li_ep_c = create_list_events(comb_ep, time_interval, time_interval * len(labels_condition), boolean_output=True)

    return li_ep_c, li_ep_p, compt_arti

def _labels_to_ep_and_bursts(labels, time_interval, delim_ep, min_burst_joining=3):
    # ------------grouping episodes and bursts together------------------------------------
    bursts = classif_to_burst(labels, time_interval=time_interval)
    li_ep = burst_to_episode(bursts, delim=delim_ep, min_burst_joining=min_burst_joining)
    events = create_list_events(li_ep, time_interval, len(labels) * time_interval, boolean_output=True)
    ep_and_bursts = np.any(np.c_[labels, events], axis=-1)  # rassembling bruxism bursts and episodes
    return ep_and_bursts, li_ep




def combine_brux_MEMA(labels_brux, labels_artifacts_brux, time_interval_brux, delim_ep_brux, labels_MEMA,
                      labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA,
                      min_burst_joining_brux=3, min_burst_joining_MEMA=0):
    """
    Hypothesis labels_artifacts_brux has the same length as labels_brux idem for MEMA

    """
    # Putting labels inputs on the same sampling and epoching
    if len(labels_brux) != len(labels_MEMA):
        if len(labels_brux) < len(labels_MEMA):
            labels_brux = merge_labels_list ([labels_brux], len(labels_MEMA))
            # adapts and fuses artifacts
            labels_artifacts =  merge_labels_list ([labels_artifacts_brux,labels_artifacts_MEMA], len(labels_MEMA))
            time_interval = time_interval_MEMA
        else:
            labels_MEMA = merge_labels_list([labels_MEMA], len(labels_brux))
            # adapts and fuses artifacts
            labels_artifacts = merge_labels_list([labels_artifacts_brux,labels_artifacts_MEMA], len(labels_brux))
            time_interval = time_interval_brux
    else: #inputs of same length
        # fuses artifacts
        labels_artifacts = merge_labels_list([labels_artifacts_brux, labels_artifacts_MEMA], len(labels_brux))
        time_interval = time_interval_brux

    # Creating lists of episode and bursts for bruxism and MEMA
    brux_burst_ep, li_ep_brux = _labels_to_ep_and_bursts(labels_brux, time_interval, delim_ep_brux,
                                                         min_burst_joining=min_burst_joining_brux)
    MEMA_burst_ep, li_ep_MEMA = _labels_to_ep_and_bursts(labels_MEMA, time_interval, delim_ep_MEMA,
                                                         min_burst_joining=min_burst_joining_MEMA)

    # Conditionnal labelling of events
    MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA= _cond_subclassif(li_ep_MEMA, labels_artifacts,
                                                                 brux_burst_ep, time_interval)
    brux_comb_ep, brux_pure_ep, compt_arti_brux = _cond_subclassif(li_ep_brux, labels_artifacts,
                                                                   MEMA_burst_ep, time_interval)

    return brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA






