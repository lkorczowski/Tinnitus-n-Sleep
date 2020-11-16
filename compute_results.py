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
    #%%
    import os
    import os.path
    import mne
    import numpy as np
    import pandas as pd
    import warnings
    from time import time
    from tinnsleep.data import read_sleep_file
    from tinnsleep.reports import reporting, generate_MEMA_report, generate_bruxism_report, preprocess
    from tinnsleep.utils import crop_to_proportional_length, resample_labels
    from tinnsleep.config import Config
    from ast import literal_eval
    from datetime import datetime

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
    OMA_extension = [1, 2] # extension of 1s of the OMA labels, put [0,0] if no extension is desired, first value is for
    #left extension and second value is for right extension
    results_file_MEMA = "data/reports_and_datas_MEMA.pk"
    results_file_bruxism = "data/reports_and_datas_bruxism.pk"

    # Dictionnary of known names of the Airflow
    print("parameters set")

    # Importing personnalized parameters for dataset
    data_info = pd.read_csv("data/data_info.csv", engine='python', sep=",")
    #data_info["Valid_chans"][data_info["Valid_chans"].isna()]='[0]'
    data_info["Valid_chans"] = data_info["Valid_chans"].apply(literal_eval)
    #data_info["Valid_imps"][data_info["Valid_imps"].isna()]='[1]'
    data_info["Valid_imps"] = data_info["Valid_imps"].apply(literal_eval)
    mema_files = data_info.query("mema == 1")["filename"].values
    dico_chans = data_info.set_index('filename')[["Valid_chans", "Valid_imps", "THR_IMP"]]

    #%% Processing of the dataset and report generation
    if (not OVERWRITE_RESULTS) and os.path.isfile(results_file_MEMA) and os.path.isfile(results_file_bruxism):
        print(f"result files exist: Reports creation skipped.")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            results_bruxism = {}
            results_MEMA = {}

            print("Files processed: ")
            start = time()
            #%%
            for filename in EDF_list: #['/Users/louis/Data/SIOPI/bruxisme/3BS04_cohort2.edf']:#

                # opens the raw file
                raw = mne.io.read_raw_edf(filename, preload=False, verbose=False)  # prepare loading
                file = filename.split(os.path.sep)[-1]
                if file == "1PI07_nuit_hab.edf":
                    print("c'est 1PI07 hab :) ! ")
                    croptimes = dict(tmin=raw.times[0] + 10750, tmax=raw.times[-1] - 3330)
                    raw.crop(**croptimes)


                print(file, end=" ")
                # Get channels indexes
                ind_picks_chan = dico_chans.loc[file]['Valid_chans']
                # Get impedance channels
                ind_picks_imp = dico_chans.loc[file]['Valid_imps']
                # Get THR_imp value for filename
                THR_imp = dico_chans.loc[file]['THR_IMP']
                # Get sleep stages if exist
                sleep_file = "data/sleep_labels/robin/" + file.split(".")[0] + ".csv"
                if os.path.isfile(sleep_file):
                    try:
                        # prepare timestamps of the sleep labels and convert it to seconds relative to beginning of
                        # recording.
                        print(f"(sleep labels", end=" ")

                        # prepare local variable
                        sleep_labels, sleep_label_timestamp = read_sleep_file(sleep_file,
                                        sep=",",
                                        encoding="ISO-8859-1",
                                        time_format='%d/%m/%Y %H:%M:%S',
                                        raw_info_start_time=raw.info["meas_date"].time())

                        print(f", loaded)", end=" ")
                    except:
                        print(f"(error with sleep labels)", end=" ")
                        sleep_labels = None
                        sleep_label_timestamp = None
                else:
                    print(f"(sleep labels not found)", end=" ")
                    sleep_labels = None
                    sleep_label_timestamp = None

                #%% ----------------- Prepare parameters -----------------------------------------------
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

                #%% ----------------- Preprocessing ----------------------------------------------------
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
                    if 'Mask Pressure' in raw.info['ch_names']:
                        raw.rename_channels({'Mask Pressure': 'Airflow', "Pression diff": 'Airflow (L)'})
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

                #%% ----------------- Finding artifacts in other channels (can be stacked) --------------
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
                    #Extending OMA episodes of OMA_extension[0] on the left and OMA_extension[1] on the right
                    valid_labels_OMA = labels_1s_extension(valid_labels_OMA, OMA_extension[0], OMA_extension[1])
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

                #%% ----------------- Merging artifacts labels ------------------------------------------

                if DO_BRUXISM:
                    epochs_bruxism, valid_labels_bruxism = crop_to_proportional_length(epochs_bruxism,
                                                                                       valid_labels_bruxism)
                if DO_MEMA:
                    epochs_MEMA, valid_labels_MEMA = crop_to_proportional_length(epochs_MEMA, valid_labels_MEMA)
                print(f"DONE ({time() - tmp:.2f}s)", end=" ")

                #%% ----------------- REPORTING ---------------------------------------------------------
                print("report... Bruxism(", end="")
                tmp = time()

                if DO_BRUXISM and np.sum(valid_labels_bruxism) > 0:
                    #
                    if sleep_labels is not None:
                        xnew = np.linspace(0, len(epochs_bruxism) * window_length_bruxism,
                                           len(epochs_bruxism), endpoint=False)
                        sleep_labels_bruxism = resample_labels(sleep_labels, xnew, x=sleep_label_timestamp,
                                                            kind='previous')
                    else:
                        sleep_labels_bruxism = None

                    results_bruxism[file] = reporting(epochs_bruxism, valid_labels_bruxism, THR_classif_bruxism,
                                                      time_interval=window_length_bruxism,
                                                      delim=delim, n_adaptive=n_adaptive_bruxism, log=log,
                                                      generate_report=generate_bruxism_report,
                                                      sleep_labels=sleep_labels_bruxism)
                    print(f"done)", end=" ")
                else:
                    print(f"skipped)", end=" ")

                print("MEMA(", end="")
                if DO_MEMA:
                    if sleep_labels is not None:
                        xnew = np.linspace(0, len(epochs_MEMA) * window_length_MEMA, len(epochs_MEMA), endpoint=False)
                        sleep_labels_MEMA = resample_labels(sleep_labels, xnew, x=sleep_label_timestamp,
                                                            kind='previous')
                    else:
                        sleep_labels_MEMA = None

                    print("report...", end="");
                    tmp = time()
                    results_MEMA[file] = reporting(epochs_MEMA, valid_labels_MEMA, THR_classif_MEMA,
                                                   time_interval=window_length_MEMA, delim=delim,
                                                   n_adaptive=n_adaptive_MEMA,
                                                   generate_report=generate_MEMA_report,
                                                   sleep_labels=sleep_labels_MEMA)
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
