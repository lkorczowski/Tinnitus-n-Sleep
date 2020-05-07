import numpy as np
import mne
from tinnsleep.config import Config
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.check_impedance import create_annotation_mne, Impedance_thresholding_sliding, check_RMS, \
    fuse_with_classif_result
from tinnsleep.signal import rms
from tinnsleep.scoring import generate_bruxism_report
from tinnsleep.signal import is_good_epochs
from scipy.interpolate import interp1d


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

    raw = CreateRaw(raw[picks_chan][0], picks_chan, ch_types='emg')  # pick channels and load

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


def merge_preprocess_labels(list_valid_labels, n_epoch_final):
    """creates a unique list of valid_labels for classification that merges artifacts coming from different sources
        ----------
        list_valid_labels : list of list of booleans
            lists obtained from different preprocessing loops.
        n_epoch_final: int
            length of the output ndarray desired
        Returns
        -------
        ndarray : ndarray of shape (n, )
            Merged valid_label list for classification
        """
    for i in range(len(list_valid_labels)):
        leny = len(list_valid_labels[i])
        if (float(leny/n_epoch_final)).is_integer() and (float(leny/n_epoch_final) != 1) :
                downsamp_factor = int(leny / n_epoch_final)
                arrival_li = [np.sum(list_valid_labels[i][j * downsamp_factor:(j + 1) * downsamp_factor])/
                              downsamp_factor == 1 for j in range(n_epoch_final)]
                arrival_li = np.asanyarray(arrival_li)

                if i == 0:
                    # adding first element to bef_merge
                    bef_merge = np.c_[arrival_li]  # f1(xnew) : projection of the data on the projection linear space
                else:
                    # Concatenating and transposing for future events
                    bef_merge = np.c_[bef_merge, arrival_li]
        else:
            # Start Linear space
            x = np.linspace(0, leny, num=leny, endpoint=True)
            # Projection linear space
            xnew = np.linspace(0, leny, num=n_epoch_final, endpoint=True)
            # Interpolation object definition
            f1 = interp1d(x, list_valid_labels[i], kind='nearest')

            if i == 0:
                # adding first element to bef_merge
                bef_merge = np.c_[f1(xnew)]  # f1(xnew) : projection of the data on the projection linear space
            else:
                # Concatenating and transposing for future events
                bef_merge = np.c_[bef_merge, f1(xnew)]



    # logical AND: if any of the preprocessing steps says the epoch is bad, it should be rejected
    print(bef_merge)
    valid_labels = np.all(bef_merge, axis=-1)
    print(valid_labels)
    return valid_labels
