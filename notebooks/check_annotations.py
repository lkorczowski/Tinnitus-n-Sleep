import pandas as pd
from tinnsleep.config import Config
import mne
from tinnsleep.data import AnnotateRaw_sliding, CleanAnnotations
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    threshold_index_bruxism = 3
    threshold_index_MEMA = 2
    data_info = pd.read_csv("data/data_info.csv", engine='python', sep=";")
    mema_files = data_info[data_info["mema"] == 1]['filename'].values
    results_MEMA = pd.read_pickle("data/reports_and_datas_MEMA.pk").to_dict()
    results_brux = pd.read_pickle("data/reports_and_datas_bruxism.pk").to_dict()
    min_burst_joining_brux = 3
    min_burst_joining_MEMA = 0
    file = '1GB18_nuit_hab.edf' #
    filename = [filename for filename in Config.bruxisme_files if filename.endswith(file)][0]
    # Loop on all the patient files
    print(file, end=" ")
    if not (file in results_brux.keys() and file in results_MEMA.keys()):
        print(results_brux[file]["parameters"]["time_interval"])
        print(f"(does not have both bruxism and mema)... skipping")
    else:
        window_length_brux = results_brux[file]["parameters"]["time_interval"]
        delim_brux = results_brux[file]["parameters"]["delim"]
        window_length_MEMA = results_MEMA[file]["parameters"]["time_interval"]
        delim_MEMA = results_MEMA[file]["parameters"]["delim"]
        params_combine = dict(
            labels_brux=results_brux[file]["labels"][threshold_index_bruxism],
            labels_artifacts_brux=results_brux[file]["parameters"]["valid_labels"],
            time_interval_brux=results_brux[file]["parameters"]["time_interval"],
            delim_ep_brux=results_brux[file]["parameters"]["delim"],
            labels_MEMA=results_MEMA[file]["labels"][threshold_index_MEMA],
            labels_artifacts_MEMA=results_MEMA[file]["parameters"]["valid_labels"],
            time_interval_MEMA=results_MEMA[file]["parameters"]["time_interval"],
            delim_ep_MEMA=results_brux[file]["parameters"]["delim"],
            min_burst_joining_brux=min_burst_joining_brux,
            min_burst_joining_MEMA=min_burst_joining_MEMA)

        raw = mne.io.read_raw_edf(filename)
        dico_chans = pd.read_pickle("data/valid_chans_THR_imp.pk").to_dict("list")[file]
        brux_channels = np.array(raw.info["ch_names"])[dico_chans[0]].tolist()
        imp_channels = np.array(raw.info["ch_names"])[dico_chans[1]].tolist()

        raw.pick_channels(ch_names=brux_channels + imp_channels + ["Airflow", "Activity"]).load_data()
        labels_brux = results_brux[file]["labels"][threshold_index_bruxism]
        labels_MEMA = results_MEMA[file]["labels"][threshold_index_MEMA]

        raw = CleanAnnotations(raw)
        duration_brux = window_length_brux * raw.info['sfreq']
        duration_MEMA = window_length_MEMA * raw.info['sfreq']
        dict_annotations = {True: "brux"}
        raw = AnnotateRaw_sliding(raw, labels_brux, dict_annotations=dict_annotations, duration=duration_brux, interval=duration_brux, merge=True)
        dict_annotations = {True: "MEMA"}
        raw = AnnotateRaw_sliding(raw, labels_MEMA, dict_annotations=dict_annotations, duration=duration_MEMA, interval=duration_MEMA, merge=True)
        dict_annotations = {True: "MEMA artefact"}
        raw = AnnotateRaw_sliding(raw, np.invert(results_MEMA[file]["parameters"]["valid_labels"]), dict_annotations=dict_annotations, duration=duration_MEMA, interval=duration_MEMA, merge=True)
        dict_annotations = {True: "Brux artefact"}
        raw = AnnotateRaw_sliding(raw, np.invert(results_brux[file]["parameters"]["valid_labels"]),
                                  dict_annotations=dict_annotations, duration=duration_brux, interval=duration_brux,
                                  merge=True)

        raw  = raw.filter(20., 99., n_jobs=2,
                          fir_design='firwin', filter_length='auto', phase='zero-double',
                          picks=brux_channels)
        matplotlib.use('TkAgg')
        raw.set_channel_types({'Airflow': 'misc', 'Activity': 'misc'})
        [raw.set_channel_types({ch: 'emg'}) for ch in brux_channels]
        [raw.set_channel_types({ch: 'bio'}) for ch in imp_channels]
        print(pd.DataFrame(results_brux[file]["log"]))
        scalings = {'emg': 5e-5, 'misc': 0.2, 'bio':1e3}
        plt.close("all")
        raw.plot(scalings=scalings)
