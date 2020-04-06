import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms, _is_good_epoch
from tinnsleep.utils import epoch

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)


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


def test_is_good_epoch(data):
    n_channels = data.shape[0]
    duration = 100
    interval = 100
    epochs = epoch(data, duration, interval, axis=1)
    epochs[2, 1, 6] = 50   # spike
    epochs[3, 0, 25] = 35  # too small spike
    epochs[0, 0, 0::] = 0  # flat channel

    channel_type_idx = dict(emg=[0, 1])
    rejection_thresholds = dict(emg=50)
    flat_thresholds = dict(emg=1e-1)
    is_good = []
    is_good_expected = [False, True, False, True]
    for epoc in epochs:
        is_good.append(_is_good_epoch(epoc, channel_type_idx=channel_type_idx,
                       rejection_thresholds=rejection_thresholds,
                       flat_thresholds=flat_thresholds))
    npt.assert_equal(is_good, is_good_expected)