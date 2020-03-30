import matplotlib.pyplot as plt
import numpy as np
from tinnsleep.data import CleanAnnotations, Annotate

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
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch (if duration == interval, no overlap)
    
    Returns
    -------
    fig: instance of matplotlib.figure.Figure
        Raw traces.
    """
    fig1 = plt.figure()
    plt.plot(detect, marker='o', markeredgecolor='r')

    raw = CleanAnnotations(raw)
    raw = Annotate(raw, labels=detect, duration=duration, interval=interval)
    fig2 = raw.plot(scalings="auto")
            
    return fig1, fig2