if __name__ == "__main__":
    import os

    PATH = os.getcwd()
    import sys

    sys.path.append(PATH + '/../')
    import mne
    import numpy as np
    import pandas as pd
    import warnings
    from time import time
    from tinnsleep.config import Config
    from tinnsleep.reports import reporting, generate_MEMA_report, generate_bruxism_report, preprocess
    from tinnsleep.utils import merge_labels_list

    print("config loaded")

    # Setting parameters
    EDF_list = Config.bruxisme_files
    THR_classif_bruxism = [[0, 2], [0, 3], [0, 4], [0, 5]]
    THR_classif_MEMA = [[0, 3], [0, 3.5], [0, 4]]

    window_length_bruxism = 0.25  # in seconds (all epoch duration will be computed from here, might not be exactly this value because of rounding)
    duration_factor_Impedance = 1  # how many time window_length
    duration_factor_MEMA = 4  # how many time window_length
    duration_factor_OMA = 4  # how many time window_length
    n_adaptive_bruxism = -120  # number of seconds for adaptive (negative for forward-backward adaptive scheme)
    n_adaptive_MEMA = -60  # number of seconds for adaptive (negative for forward-backward adaptive scheme)
    delim = 3  # maximal time interval between bursts to merge episode in seconds
    results_file_MEMA = "data/reports_and_datas_MEMA.pk"
    results_file_bruxism = "data/reports_and_datas_bruxism.pk"

    OVERWRITE_RESULTS = False  # re-compute the results even if files exits

    # Dictionnary of known names of the Airflow
    mapping = {"Airflow": "MEMA"}
    print("parameters set")

    # Importing personnalized parameters for dataset
    mema_files = pd.read_csv("data/mema_files.csv", engine='python', sep="; ")["files_with_mema"].values
    dico_chans = pd.read_pickle("data/valid_chans_THR_imp.pk").to_dict("list")

    ## Processing of the dataset and report generation

    if not OVERWRITE_RESULTS and os.path.isfile(results_file_MEMA) and os.path.isfile(results_file_bruxism):
        print(f"result files exist: Reports creation skipped.")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            results_bruxism = {}
            results_MEMA = {}

            print("Files processed: ")
            start = time()
            for filename in EDF_list:

                # opens the raw file
                raw = mne.io.read_raw_edf(filename, preload=False, verbose=False)  # prepare loading
                file = filename.split(os.path.sep)[-1]

                print(file, end=" ")
                # Get channels indexes
                ind_picks_chan = dico_chans[file][0]
                ind_picks_imp = dico_chans[file][1]
                # Get THR_imp value for filename
                THR_imp = dico_chans[file][2]

                # Get channel names from indexes
                if len(ind_picks_chan) > 0:  # ignore file if no channel is good
                    # ----------------- Prepare parameters -----------------------------------------------
                    duration_bruxism = int(window_length_bruxism * raw.info['sfreq'])  # in sample
                    window_length_bruxism = duration_bruxism / raw.info['sfreq']  # recompute exact window_length
                    duration_MEMA = duration_factor_MEMA * duration_bruxism
                    window_length_MEMA = duration_MEMA / raw.info['sfreq']  # recompute exact window_length
                    duration_OMA = duration_factor_OMA * duration_bruxism
                    duration_Impedance = duration_factor_Impedance * duration_bruxism
                    n_adaptive_bruxism = int(n_adaptive_bruxism / window_length_bruxism)

                    log = {}

                    picks_chan_bruxism = []
                    for elm in ind_picks_chan:
                        picks_chan_bruxism.append(raw.info["ch_names"][elm])

                    picks_imp = []
                    for elm in ind_picks_imp:
                        picks_imp.append(raw.info["ch_names"][elm])

                    # ----------------- EMG processing ---------------------------------------------------
                    print("preprocess...", end="")
                    tmp = time()
                    is_good_kwargs = dict(ch_names=picks_chan_bruxism,
                                          rejection_thresholds=dict(emg=5e-05),  # two order of magnitude higher q0.01
                                          flat_thresholds=dict(emg=1e-09),  # one order of magnitude lower median
                                          channel_type_idx=dict(emg=[i for i in range(len(picks_chan_bruxism))])
                                          )
                    epochs_bruxism, valid_labels_bruxism, log["bruxism"] = preprocess(raw, duration_bruxism,
                                                                                      duration_bruxism,
                                                                                      picks_chan=picks_chan_bruxism,
                                                                                      is_good_kwargs=is_good_kwargs)
                    valid_labels_bruxism = [valid_labels_bruxism]

                    # ----------------- MEMA processing ---------------------------------------------------
                    epochs_MEMA, valid_labels_MEMA, log["MEMA"] = preprocess(raw, duration_MEMA, duration_MEMA,
                                                                             picks_chan=['MEMA'])
                    valid_labels_MEMA = [valid_labels_MEMA]

                    # ----------------- Finding artifacts in other channels (can be stacked) --------------
                    # 1. OMA
                    _, valid_labels_OMA, log["OMA"] = preprocess(raw, duration_OMA, duration_OMA,
                                                                 picks_chan=["Activity"],
                                                                 Thresholding_kwargs=dict(abs_threshold=0,
                                                                                          rel_threshold=3,
                                                                                          decision_function=lambda
                                                                                              foo: np.any(
                                                                                              foo > 0, axis=-1)),
                                                                 burst_to_episode_kwargs=dict(min_burst_joining=0,
                                                                                              delim=3)
                                                                 )
                    valid_labels_bruxism.append(valid_labels_OMA)
                    valid_labels_MEMA.append(valid_labels_OMA)

                    # 2. Impedance Check
                    _, valid_labels_IMP, log["IMP"] = preprocess(raw, duration_Impedance, duration_Impedance,
                                                                 picks_chan=picks_imp,
                                                                 Thresholding_kwargs=dict(abs_threshold=THR_imp,
                                                                                          rel_threshold=0,
                                                                                          decision_function=lambda
                                                                                              foo: np.any(
                                                                                              foo > 0, axis=-1)),
                                                                 burst_to_episode_kwargs=dict(min_burst_joining=0,
                                                                                              delim=1)
                                                                 )
                    valid_labels_bruxism.append(valid_labels_IMP)

                    # ----------------- Merging artifacts labels ------------------------------------------
                    def crop_to_proportional_length(epochs, valid_labels):
                        """align number of epochs and a list of valid_labels to be proportional"""
                        # compute all resampling factors
                        resampling_factors = [int(len(epochs) / len(i)) for i in valid_labels]

                        # find the common denominator
                        min_labels = min([len(i) * j for (i, j) in zip(valid_labels, resampling_factors)])
                        assert (len(epochs) - min_labels) < max(
                            resampling_factors), f"shift of {len(epochs) - min_labels} epochs, please check that all duration are proportional"
                        epochs = epochs[:min_labels]  # crop last epochs
                        valid_labels_crop = [i[:int(min_labels / j)] for (i, j) in
                                                zip(valid_labels, resampling_factors)]  # crop valid_labels
                        assert len(epochs_bruxism) == min_labels, f"something went wrong when cropping"
                        valid_labels_crop = merge_labels_list(valid_labels_crop, len(epochs))
                        return epochs, valid_labels_crop


                    epochs_bruxism, valid_labels_bruxism = crop_to_proportional_length(epochs_bruxism, valid_labels_bruxism)
                    epochs_MEMA, valid_labels_MEMA = crop_to_proportional_length(epochs_MEMA, valid_labels_MEMA)

                    print(f"DONE ({time() - tmp:.2f}s)", end=" ")

                    # ----------------- Bruxism reporting -------------------------------------------------
                    print("report... Bruxism(", end="")
                    tmp = time()

                    if np.sum(valid_labels_bruxism) > 0:
                        results_bruxism[file] = reporting(epochs_bruxism, valid_labels_bruxism, THR_classif_bruxism,
                                                          time_interval=window_length_bruxism,
                                                          delim=delim, n_adaptive=n_adaptive_bruxism, log=log,
                                                          generate_report=generate_bruxism_report)
                        print(f"done)", end=" ")
                    else:
                        print(f"skipped)", end=" ")

                    #----------------- MEMA reporting ----------------------------------------------------
                    print("MEMA(", end="")

                    if np.sum(valid_labels_MEMA)>0 :
                        print("report...", end="");tmp = time()
                        results_MEMA[filename] = reporting(epochs_MEMA, valid_labels_MEMA, THR_classif_MEMA,
                                                      time_interval=window_length_MEMA, delim=delim,
                                                      n_adaptive=n_adaptive_MEMA,
                                                      generate_report=generate_MEMA_report)
                        print(f"done)", end=" ")
                    else:
                        print(f"skipped)", end=" ")
                    print(f"DONE ({time() - tmp:.2f}s)")

        print(f"Reports created, process finished in {(time() - start) / 60:.1f} min")
        pd.DataFrame.from_dict(results_bruxism).to_pickle(results_file_bruxism)
        pd.DataFrame.from_dict(results_MEMA).to_pickle(results_file_MEMA)
        print(f"Results saved")
