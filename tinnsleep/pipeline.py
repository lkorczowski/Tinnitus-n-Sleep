from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.signal import rms
from tinnsleep.events.scoring import classif_to_burst, burst_to_episode, episodes_to_list
import numpy as np


def forward_backward_AmplitudeTresholding(epochs,
                                          window_length,
                                          length_adaptive=60,
                                          relative_threshold=3.5,
                                          max_duration_between_bursts=3
                                          ):

    # -----------------MEMA processing Forward-backward ---------------------------------------
    # Foward
    # compute the sum of power over electrodes and samples in each window
    pipeline = AmplitudeThresholding(abs_threshold=0., rel_threshold=relative_threshold, n_adaptive=length_adaptive)
    X = rms(epochs)  # take only valid labels
    labels_f = pipeline.fit_predict(X)

    # Backward
    # Reversing epochs array
    epochs = epochs[::-1]
    # compute the sum of power over electrodes and samples in each window
    pipeline = AmplitudeThresholding(abs_threshold=0., rel_threshold=relative_threshold, n_adaptive=length_adaptive)
    X = rms(epochs)  # take only valid labels
    labels = pipeline.fit_predict(X)
    # Reversing labels
    labels_b = labels[::-1]

    # -----------------MEMA foward-backward merge ---------------------------------------
    # Logical OR -- merged backward and foward
    labels_fb = np.any(np.c_[labels_f, labels_b], axis=-1)

    # -----------------MEMA bursts conversion to episodes ----------------------------------------
    list_bursts = classif_to_burst(labels_fb, time_interval=window_length)
    list_episodes = burst_to_episode(list_bursts, delim=max_duration_between_bursts, min_burst_joining=0)
    list_labels = episodes_to_list(list_episodes, window_length, len(labels_fb))

    return list_labels
