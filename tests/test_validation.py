import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.visualization import (plotTimeSeries,
                                     assert_y_labels_correct)
from tinnsleep.validation import assert_ax_equals_data, is_valid_ch_names



def test_asserts_homemade():
    """Check if homemade asserts catch the error"""
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data)

    # check if x-labels are incorrect values
    with pytest.raises(AssertionError, match=r"Arrays are not equal\n\nMismatched elements: 399 / 400 (99.8%)*"):
        # check if correct values
        assert_ax_equals_data(data, ax, sfreq=2)

    # check if y-label values are incorrect
    with pytest.raises(AssertionError, match="labels are not the same"):
        assert_y_labels_correct(data, ["0" for k in range(data.shape[1])])

    np.random.seed(10)
    data = np.random.randn(400, 2)

    # check if y-label position are incorrect
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_y_labels_correct(data, ["0" for k in range(data.shape[1])])

    # check if data values are incorrect values
    with pytest.raises(AssertionError, match=r"Items are not equal to 7 significant digits:"):
        # check if correct values
        assert_ax_equals_data(data, ax, sfreq=sfreq)


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
