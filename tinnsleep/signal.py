import numpy as np

def rms(epochs, axis=-1):
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode over the samples.

    .. math::

        rms = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(x_{i})^{2}}

    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_electrodes, n_samples)
        the epochs for the estimation
    axis : None or int or tuple of ints, optional (default: -1)
        Axis or axes along which the means are computed.

    Returns
    -------
    RMS : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    """

    return np.sqrt(np.mean(epochs ** 2, axis=axis))
