import numpy as np
import mne
from tinnsleep.config import Config
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.check_impedance import create_annotation_mne, Impedance_thresholding_sliding, check_RMS, \
    fuse_with_classif_result
from tinnsleep.signal import rms
from tinnsleep.scoring import generate_clinical_report
from tinnsleep.signal import is_good_epochs


def create_reports(EDF_list, ind_picks_chan, ind_picks_imp, THR_classif, duration, interval, THR_imp=6000):
    """creates clinical reports of bruxism out of a list of edf files, for different thresholding values
        Parameters
        ----------
        EDF_list : list of str
            Names of the recordings to process as named in the folder to browse
        picks_chan: array-like, shape (n_channels,)
            list of INDEXES corresponding of the name of each channel to analyze
        picks_imp: array-like, shape (n_channels,)
            list of INDEXES corresponding of the name of each channel of impedance values associated with the channels to
            analyze
        THR_classif : list of floats
            list of the threshold values of the classifier to test
        duration : int
            Number of elements (i.e. samples) on the epoch.
        interval: int
            Number of elements (i.e. samples) to move for the next epoch.
        THR_imp: float
            Threshold value for the impedance rejection algorithm
        Returns
        -------

        results dictionary of dictionaries
            Dictionary where each key is a patient and each value for each patient is a dictionary containing fields:
            - THR_classif: threshold(s) of classification tested
            - labels : list of labels of bursts for the epochs of the recording for each THR_classif in the same order
            - reports: list of clinical reports of the recording for each THR_classif tested in the same order
            - log : log of the pre-processing operations
        """

    results = {}
    for filename in EDF_list:

        # Opening the file
        raw = mne.io.read_raw_edf(filename, preload=False)  # prepare loading
        # Default value for picks_imp
        picks_imp = []
        picks_chan = []
        ch_names = raw.info["ch_names"]
        for elm in ind_picks_imp:
            picks_imp.append(ch_names[elm])
        for elm in ind_picks_chan:
            picks_chan.append(ch_names[elm])

        raw = CreateRaw(raw[picks_chan][0], picks_chan, ch_types=['emg'])  # pick channels and load
        # Filtering data
        raw = raw.filter(20., 99., n_jobs=4,
                         fir_design='firwin', filter_length='auto', phase='zero-double',
                         picks=picks_chan)

        # Creating epochs
        epochs = RawToEpochs_sliding(raw, duration=duration, interval=interval)

        # Epoch rejection based on |min-max| thresholding
        params = dict(ch_names=raw.info["ch_names"],
                      rejection_thresholds=dict(emg=1e-04),  # two order of magnitude higher q0.01
                      flat_thresholds=dict(emg=1e-09),  # one order of magnitude lower median
                      channel_type_idx=dict(emg=[0, 1]),
                      full_report=True
                      )
        amplitude_labels, bad_lists = is_good_epochs(epochs, **params)
        suppressed_amp = np.sum(np.invert(amplitude_labels))

        # Epoch rejection based on impedance
        raw_imp = mne.io.read_raw_edf(filename, preload=False)  # prepare loading
        check_imp = Impedance_thresholding_sliding(raw_imp[picks_imp][0], duration, interval, THR=THR_imp)
        impedance_labels = np.any(check_imp, axis=-1)
        suppressed_imp = np.sum(impedance_labels)

        # Reuniting the rejection algorithms
        valid_labels = np.any(np.c_[np.invert(impedance_labels), amplitude_labels], axis=-1)  # Logical OR
        suppressed_all = np.sum(np.invert(valid_labels))

        labs = []
        reps = []
        # for each value of THR_classif, create a report and a list of labels
        for THR in THR_classif:
            pipeline = AmplitudeThresholding(abs_threshold=0., rel_threshold=THR)
            X = rms(epochs[valid_labels])  # take only valid labels
            labels = pipeline.fit_predict(X)
            report = generate_clinical_report(labels)
            labels = fuse_with_classif_result(np.invert(valid_labels),
                                              labels)  # add the missing labels removed with artefacts
            labs.append(labels)
            reps.append(report)

        # Adding the result for each patient
        results[filename] = {"THR_classif": THR_classif, "labels": labs, "reports": reps,
                             "log": {"suppressed_imp_THR": suppressed_imp, "suppressed_amp_THR": suppressed_amp,
                                     "suppressed_overall": suppressed_all}}

    return results
