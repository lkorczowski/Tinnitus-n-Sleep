import numpy as np
from tinnsleep.classification import AmplitudeThresholding


def rms(epochs, axis=2):
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode.

    .. math::

        rms = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(x_{i})^{2}}

    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_electrodes, n_samples)
        the epochs for the estimation

    Returns
    -------
    RMS : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    """

    return np.sqrt(np.mean(epochs ** 2, axis=axis))