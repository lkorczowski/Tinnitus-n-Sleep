from tinnsleep.config import Config
import mne
from tinnsleep.data import align_labels_with_raw
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import warnings


if __name__ == "__main__":
    EDF_list = Config.bruxisme_files
    labels_dict = {}
    labels_timestamp_dict = {}
    df_results = pd.DataFrame()

    for filename in EDF_list:
        file = filename.split(os.path.sep)[-1]
        print(f"{file}:", end=" ")
        sleep_files = []
        # zip, kind, file and separator
        sleep_files.append(("reference", "data/sleep_labels/" + file.split(".")[0] + ".csv", ";"))
        sleep_files.append(("noxturnal", "data/sleep_labels/auto_noxturnal/" + file.split(".")[0] + ".csv", ","))
        #sleep_files.append(("algo", "data/sleep_labels/auto_algo/" + file.split(".")[0] + ".csv", ""))
        all_exists = [os.path.isfile(x) for _, x, _ in sleep_files]
        if np.all(all_exists):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_edf(filename)
            # prepare timestamps of the sleep labels and convert it to seconds relative to beginning of
            # recording.
            print(f"sleep labels...", end="")

            map_columns = {"Horodatage": "Start Time", "Sommeil": "Sleep", "event": "Sleep", "begin": "Start Time"}
            labels_dict[file] = []
            labels_timestamp_dict[file] = []
            for kind, sleep_file, sep in sleep_files:
                try:
                    sleep_labels = pd.read_csv(sleep_file, sep=sep, encoding="ISO-8859-1")
                    sleep_labels = sleep_labels.rename(columns=map_columns)
                    sleep_label_timestamp = align_labels_with_raw(sleep_labels["Start Time"],
                                          raw.info["meas_date"].time(), raw.times)
                    sleep_labels = sleep_labels["Sleep"].values

                    # replace specific values
                    sleep_labels[sleep_labels=='\x83veil'] = 'Wake'
                    sleep_labels[sleep_labels=='AWA'] = 'Wake'
                    sleep_labels[sleep_labels.astype(str)=='nan'] = 'Wake'

                    if kind != 'reference':
                        if labels_timestamp_dict[file][0][0] == 'reference':
                            sleep_label_timestamp_reference = labels_timestamp_dict[file][0][1]
                            intersectionT, indtref, indt = np.intersect1d(sleep_label_timestamp_reference, sleep_label_timestamp,
                                                     assume_unique=True, return_indices=True)
                            # sleep_labels = resample_labels(sleep_labels,
                            #                 labels_timestamp_dict[file][0][1],
                            #                 x=sleep_label_timestamp,
                            #                 kind='previous')
                            y_predicted = sleep_labels[indt]
                            y = labels_dict[file][0][1][indtref]
                            unique_labels = np.unique(y)

                            df = pd.DataFrame.from_dict(classification_report(y, y_predicted, labels=unique_labels, output_dict=True)
                                                   ).transpose()
                            df['accuracy'] = np.nan  # initialize column
                            for stage in unique_labels:
                                dfnew = pd.DataFrame({'accuracy': [accuracy_score(y == stage, y_predicted == stage)]},
                                             index=[stage])
                                df.update(dfnew)
                            df['file'] = file
                            df['kind'] = kind
                            df_results = df_results.append(df)

                    labels_dict[file].append((kind, sleep_labels))
                    labels_timestamp_dict[file].append((kind, sleep_label_timestamp))
                except ValueError:
                    labels_dict[file].append((kind, None))
                    labels_timestamp_dict[file].append((kind, None))

        else:
            print("missing sleep labels")
    df_results.to_csv("./data/results_sleep_comparison.csv")
