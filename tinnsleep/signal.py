import numpy as np
from tinnsleep.classification import AmplitudeThresholding


def rms(epochs, ax=1):
    """ Estimate Root Mean Square Amplitude for each epoch and each electrode.

    .. math::

        rms = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(x_{i})^{2}}

    Parameters
    ----------
    epochs : ndarray, shape (n_trials, n_samples, n_electrodes)
        the epochs for the estimation

    Returns
    -------
    RMS : ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude
    """

    return np.sqrt(np.mean(epochs ** 2, axis=ax))

def create_basic_detection(RMS, absv, relv):
    """ Create an array of booleans with True corresponding to an epoch classified
    as containing potentially a bruxism burst
    
    Parameters
    ----------
    RMS: ndarray, shape (n_trials, n_electrodes)
        Root Mean Square Amplitude of a series of epochs
    
    Returns
    -------
    l_detect : ndarray, shape (n_trials)
        binary array output of the classification process
    """
    #Create a new instance of AmplitudeThresholding
    ampthr=AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv) 
    #returns the prediction of the classifier
    return ampthr.fit_predict(RMS)
    
    
    
    
    
    
        
    