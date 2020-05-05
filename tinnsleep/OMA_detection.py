import mne
import numpy as np
from tinnsleep.utils import epoch
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.signal import rms


# Create Raw file
def OMA_thresholding_sliding(data, duration, interval, OMA_THR=[0,3], OMA_adap=0):
    """Tags each electrode of each epoch of a recording with a label, True meaning fine, False meaning Bad epoch for this
    channel
    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    duration: int
        Number of elements (i.e. samples) on the epoch.
    interval: int
        Number of elements (i.e. samples) to move for the next epoch.
    OMA_THR : list of 2 floats
        absolute and relative thresholds for amplitude thresholding.

    Returns
    -------
    OMA_labels list of booleans of length nb_epochs
        False marking for bad epochs
    """
    # Epoching of the data
    epochs = epoch(data, duration, interval)
    pipeline = AmplitudeThresholding(abs_threshold=OMA_THR[0], rel_threshold=OMA_THR[1], n_adaptive=OMA_adap)
    X = rms(epochs)  # take only valid labels
    OMA_labels = np.invert(pipeline.fit_predict(X)) #invertion necessary so as spotted epochs get "False" labels.
    return OMA_labels

