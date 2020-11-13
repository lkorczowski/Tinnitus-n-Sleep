from tinnsleep.config import Config
import mne
from tinnsleep.data import align_labels_with_raw, read_sleep_file
from tinnsleep.utils import round_time
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    EDF_list = Config.bruxisme_files
    labels_dict = {}
    labels_timestamp_dict = {}
    df_results = pd.DataFrame()

    for filename in EDF_list:
        file = filename.split(os.path.sep)[-1]
        print(f"{file}:", end=" ")
        sleep_files = []  # TUPPLE LIST FORMAT : (kind, file, separator, time format)

        # ALWAYS APPEND reference labels first
        sleep_files.append(("reference", "data/sleep_labels/maxime/" + file.split(".")[0] + ".csv", ";", '%H:%M:%S'))

        # APPEND the labels you want to compare with the reference labels
        sleep_files.append(("noxturnal", "data/sleep_labels/auto_noxturnal/" + file.split(".")[0] + ".csv", ",", '%H:%M:%S'))
        sleep_files.append(("algo", "data/sleep_labels/auto_algo/" + file.split(".")[0] + ".csv", ";", '%Y-%m-%d %H:%M:%S'))

        all_exists = [os.path.isfile(x) for _, x, _, _ in sleep_files]
        if np.all(all_exists):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_edf(filename, verbose='ERROR')
            # prepare timestamps of the sleep labels and convert it to seconds relative to beginning of
            # recording.
            print(f"sleep labels...", end="")

            # add any csv's column name to fit to "Start Time" and "Sleep" if different.
            map_columns = {"Horodatage": "Start Time",
                           "Sommeil": "Sleep",
                           "event": "Sleep",
                           "begin": "Start Time"}
            labels_dict[file] = []
            labels_timestamp_dict[file] = []
            for kind, sleep_file, sep, time_format in sleep_files:
                try:
                    # extract sleep labels and timestamp
                    sleep_labels, sleep_label_timestamp = read_sleep_file(sleep_file,
                                    map_columns=map_columns,
                                    sep=sep,
                                    encoding="ISO-8859-1",
                                    time_format=time_format,
                                    raw_info_start_time=raw.info["meas_date"].time(),
                                    raw_times=raw.times)

                    # round to closest 30 sec.
                    f_round = lambda x: np.ceil(x / 30).astype(int)*30
                    sleep_label_timestamp = f_round(sleep_label_timestamp)

                    # compare all labels to reference
                    if kind != 'reference':

                        if labels_timestamp_dict[file][0][0] == 'reference': # TODO: this condition is just a failsafe


                            sleep_label_timestamp_reference = labels_timestamp_dict[file][0][1]

                            # intersect1d takes only the timestamp that are perfectly aligned
                            intersectionT, indtref, indt = np.intersect1d(sleep_label_timestamp_reference, sleep_label_timestamp,
                                                     assume_unique=True, return_indices=True)

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
            print("loaded")

        else:
            print("missing sleep labels")
    print("saving results")
    df_results.to_csv("./data/results_sleep_comparison.csv")

    plt.subplot(141)
    ax = sns.boxplot(x="kind", y="recall", data=df_results.loc["Wake"])
    plt.subplot(142)
    ax = sns.boxplot(x="kind", y="precision", data=df_results.loc["Wake"])
    plt.subplot(143)
    ax = sns.boxplot(x="kind", y="precision", data=df_results.loc["Wake"])
    plt.subplot(144)
    ax = sns.boxplot(x="kind", y="f1-score", data=df_results.loc["Wake"])
    plt.savefig("./data/results_sleep_comparison_wake.png")

    plt.show()
