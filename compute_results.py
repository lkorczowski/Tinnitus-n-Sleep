import sys, getopt


def main(argv):
    bruxism = False
    mema = False
    overwrite = False
    try:
        opts, args = getopt.getopt(argv, "hb:m:o:", ["bruxism=", "mema=", "overwrite="])
    except getopt.GetoptError:
        print(f'compute_results.py --bruxism <boolean> --mema <boolean> --overwrite <boolean>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(f'compute_results.py --bruxism <boolean> --mema <boolean>')
            sys.exit()
        elif opt in ("-b", "--bruxism"):
            bruxism = arg == 'True'
        elif opt in ("-m", "--mema"):
            mema = arg == 'True'
        elif opt in ("-o", "--overwrite"):
            overwrite = arg == 'True'  # re-compute the results even if files exits
    print(f'Performs Bruxism: <{bruxism}>')
    print(f'Performs MEMA: <{mema}>')
    print(f'Will overwrite existing results: <{overwrite}>')

    return bruxism, mema, overwrite


print("config loaded.")

if __name__ == "__main__":
    import os
    import os.path
    import mne
    import numpy as np
    import pandas as pd
    import warnings
    from time import time
    from tinnsleep.reports import reporting, generate_MEMA_report, generate_bruxism_report, preprocess
    from tinnsleep.utils import crop_to_proportional_length
    from tinnsleep.config import Config

    bruxism, mema, OVERWRITE_RESULTS = main(sys.argv[1:])
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks/"))
    # Setting parameters
    EDF_list = Config.bruxisme_files
    THR_classif_bruxism = [[0, 2], [0, 3], [0, 4], [0, 5]]
    THR_classif_MEMA = [[0, 3], [0, 3.5], [0, 4]]

    window_length_common = 0.25  # in seconds (all epoch duration will be computed from here, might not be exactly this value because of rounding)
    duration_factor_Impedance = 1  # how many time window_length
    duration_factor_MEMA = 4  # how many time window_length
    duration_factor_OMA = 4  # how many time window_length
    n_adaptive_bruxism = -120  # number of seconds for adaptive (negative for forward-backward adaptive scheme)
    n_adaptive_MEMA = -60  # number of seconds for adaptive (negative for forward-backward adaptive scheme)
    delim = 3  # maximal time interval between bursts to merge episode in seconds
    results_file_MEMA = "data/reports_and_datas_MEMA.pk"
    results_file_bruxism = "data/reports_and_datas_bruxism.pk"

    # Dictionnary of known names of the Airflow
    mapping = {"Airflow": "MEMA"}
    print("parameters set")

    # Importing personnalized parameters for dataset
    mema_files = pd.read_csv("data/mema_files.csv", engine='python', sep="; ")["files_with_mema"].values
    dico_chans = pd.read_pickle("data/valid_chans_THR_imp.pk").to_dict("list")  # TODO: check if valid for all subjects

    # Processing of the dataset and report generation

    if (not OVERWRITE_RESULTS) and os.path.isfile(results_file_MEMA) and os.path.isfile(results_file_bruxism):
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
                if file == "1PI07_nuit_hab.edf":
                    print("c'est 1PI07 hab :) ! ")
                    croptimes = dict(tmin=raw.times[0] + 10750, tmax=raw.times[-1] - 3330)
                    raw.crop(**croptimes)


                print(file, end=" ")
                # Get channels indexes
                ind_picks_chan = dico_chans[file][0]
                ind_picks_imp = dico_chans[file][1]
                # Get THR_imp value for filename
                THR_imp = dico_chans[file][2]

                # ----------------- Prepare parameters -----------------------------------------------
                duration_bruxism = int(window_length_common * raw.info['sfreq'])  # in sample
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

                DO_BRUXISM = len(picks_chan_bruxism) > 0 and bruxism
                DO_MEMA = file in mema_files and mema

                # ----------------- Preprocessing ----------------------------------------------------
                print(f"preprocess...", end=" ")
                valid_labels_bruxism = []
                valid_labels_MEMA = []

                print(f"Bruxism(", end="")
                tmp = time()
                if DO_BRUXISM:
                    is_good_kwargs = dict(ch_names=picks_chan_bruxism,
                                          rejection_thresholds=dict(emg=7e-04),  # two order of magnitude higher q0.01
                                          flat_thresholds=dict(emg=3e-09),  # one order of magnitude lower median
                                          channel_type_idx=dict(emg=[i for i in range(len(picks_chan_bruxism))])
                                          )
                    filter_kwargs = dict(l_freq=20., h_freq=99., n_jobs=4,
                                         fir_design='firwin', filter_length='auto', phase='zero-double',
                                         picks=picks_chan_bruxism)
                    epochs_bruxism, valid_labels, log["bruxism"] = preprocess(raw, duration_bruxism,
                                                                              duration_bruxism,
                                                                              picks_chan=picks_chan_bruxism,
                                                                              is_good_kwargs=is_good_kwargs,
                                                                              filter_kwargs=filter_kwargs)
                    valid_labels_bruxism.append(valid_labels)
                    print(f"done)", end=" ")
                else:
                    print(f"skipped)", end=" ")

                print(f"MEMA(", end="")
                if DO_MEMA:
                    filter_kwargs = dict(l_freq=0.25, h_freq=16., n_jobs=4,
                                         fir_design='firwin', filter_length='auto', phase='zero-double',
                                         picks=['Airflow'])
                    epochs_MEMA, valid_labels, log["MEMA"] = preprocess(raw, duration_MEMA, duration_MEMA,
                                                                        picks_chan=['Airflow'],
                                                                        filter_kwargs=filter_kwargs)
                    valid_labels_MEMA.append(valid_labels)
                    print(f"done)", end=" ")
                else:
                    print(f"skipped)", end=" ")

                # ----------------- Finding artifacts in other channels (can be stacked) --------------
                # 1. OMA
                if DO_BRUXISM or DO_MEMA:
                    filter_kwargs = dict(l_freq=0.25, h_freq=16., n_jobs=4,
                                         fir_design='firwin', filter_length='auto', phase='zero-double',
                                         picks=['Activity'])
                    _, valid_labels_OMA, log["OMA"] = preprocess(raw, duration_OMA, duration_OMA,
                                                                 picks_chan=["Activity"],
                                                                 filter_kwargs=filter_kwargs,
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
                if DO_BRUXISM:

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

                if DO_BRUXISM:
                    epochs_bruxism, valid_labels_bruxism = crop_to_proportional_length(epochs_bruxism,
                                                                                       valid_labels_bruxism)
                if DO_MEMA:
                    epochs_MEMA, valid_labels_MEMA = crop_to_proportional_length(epochs_MEMA, valid_labels_MEMA)
                print(f"DONE ({time() - tmp:.2f}s)", end=" ")

                # ----------------- REPORTING ---------------------------------------------------------
                print("report... Bruxism(", end="")
                tmp = time()

                if DO_BRUXISM and np.sum(valid_labels_bruxism) > 0:
                    results_bruxism[file] = reporting(epochs_bruxism, valid_labels_bruxism, THR_classif_bruxism,
                                                      time_interval=window_length_bruxism,
                                                      delim=delim, n_adaptive=n_adaptive_bruxism, log=log,
                                                      generate_report=generate_bruxism_report)
                    print(f"done)", end=" ")
                else:
                    print(f"skipped)", end=" ")

                print("MEMA(", end="")
                if DO_MEMA:
                    print("report...", end="");
                    tmp = time()
                    results_MEMA[file] = reporting(epochs_MEMA, valid_labels_MEMA, THR_classif_MEMA,
                                                   time_interval=window_length_MEMA, delim=delim,
                                                   n_adaptive=n_adaptive_MEMA,
                                                   generate_report=generate_MEMA_report)
                    print(f"done)", end=" ")
                else:
                    print(f"skipped)", end=" ")

                print(f"DONE ({time() - tmp:.2f}s)")

        print(f"Reports created, process finished in {(time() - start) / 60:.1f} min")
        if bruxism:
            pd.DataFrame.from_dict(results_bruxism).to_pickle(results_file_bruxism, protocol=3)
        if mema:
            pd.DataFrame.from_dict(results_MEMA).to_pickle(results_file_MEMA, protocol=3)
        print(f"Results saved")
