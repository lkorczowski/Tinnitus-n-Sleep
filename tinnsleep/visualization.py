import matplotlib.pyplot as plt

def plotTimeSeries(data, fig=None,
                   ch_names=None,
                   sfreq=None,
                   scalings=None,
                   fontsize=12,
                   annotations=None,
                   **kwargs):
    """Return the exact number of expected windows based on the samples (N), window_length (T) and interval (I)

    Parameters
    ----------
    N: int
        total number of samples of the full signal to be epoched
    T: int
        duration of each epoch in samples
    I: int
        interval between the start of each epoch (if T == I, no overlap), always > 0

    Returns
    -------
    n_epochs: int
        estimated number of epochs
    """

    plt.figure()

    return fig