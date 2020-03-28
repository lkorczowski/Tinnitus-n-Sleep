import matplotlib.pyplot as plt
import numpy as np


def create_visual(raw, detect, duration, interval):
    """ Create two visuals of a burst detection : a simple plot enabling 
    visualization of time repartition and a display over the raw signal as annotations.
    Note: any existing annotation of the raw instance will be removed.
    
    Parameters
    ----------
    raw : instance of mne.Raw
            the signal of interest
    detect: ndarray, shape (n_trials,)
            the vector of labels for the annotation sampled at the interval rate
    duration : float
            duration in seconds of an epoch
    interval : float
            interval between two epochs enabling overlap (if duration == interval, no overlap)
    
    Returns
    -------
    fig: instance of matplotlib.figure.Figure
        Raw traces.
    """
    fig1 = plt.figure()
    plt.plot(detect, marker='o', markeredgecolor='r')

    if len(raw.annotations)>0:
        raw.annotations.delete(np.arange(0, len(raw.annotations)))
    for i in range(len(detect)):
        if detect[i]:
            raw.annotations.append([interval * i], [duration], str("brux_burst"))
    fig2 = raw.plot(scalings="auto")
            
    return fig1, fig2
