import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms, is_good_epoch, is_valid_ch_names


def test_rms():
    X = np.array([
        [[1, 2, 3],
         [1, 2, 3]],
        [[1, 3, 3],
         [2, 4, 10]],
        [[0, 0, 1],
          [1, 0, 0]]
    ])
    rms_values = rms(X)
    npt.assert_almost_equal(rms_values, np.array([[2.1602469,  2.1602469 ],[2.51661148, 6.32455532],[0.57735027, 0.57735027]]), decimal=4)

def test_is_valid_ch_names():
    """this compact test suite validate the is_valid_ch_names that should be used in functions to test this parameters"""
    npt.assert_equal(is_valid_ch_names([], 2), [0, 1])  # valid
    npt.assert_equal(is_valid_ch_names(None, 2), [0, 1])  # valid
    npt.assert_equal(is_valid_ch_names(["Fz", "Cz"], 2), ["Fz", "Cz"])  # valid
    with pytest.raises(ValueError, match=r"`ch_names` should be same length as the number of channels of data"):
        is_valid_ch_names(["Fz", "Cz", "Pz"], 2)   # invalid
    npt.assert_equal(is_valid_ch_names("eeg", 3), ["eeg"] * 3)  # valid
    with pytest.raises(ValueError, match=r"`ch_names` must be a list or an iterable of shape \(n_channels,\) or None"):
        is_valid_ch_names(dict(test=0), 2)   # invalid





