import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms, is_good_epochs, _is_good_epoch, power_ratio, get_peaks
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
    npt.assert_almost_equal(rms_values,
                            np.array([[2.1602469, 2.1602469], [2.51661148, 6.32455532], [0.57735027, 0.57735027]]),
                            decimal=4)

def test_get_peaks():
    epochs = np.array([[[0,0,0.5,0,1, 2,1 ,0]]])

    print(get_peaks(epochs))
    npt.assert_equal(get_peaks(epochs), [[(np.array([2, 5]), {}),(np.array([0.5, 2. ]),
                                                                                           np.array([1, 3]
                                                            ), np.array([3, 7]))]])



def test_power_ratio():
    epochs = np.random.randn(2000, 2, 2000)
    labels = [True]*1000 + [False]*1000
    npt.assert_allclose(power_ratio(epochs, labels), [1, 1], rtol=1e-2)


def test_power_ratio_missing():
    epochs = np.random.randn(2000, 2, 2000)
    labels = [True]*2000
    npt.assert_equal( power_ratio(epochs, labels), [np.nan, np.nan])

    labels = [False]*2000
    npt.assert_equal(power_ratio(epochs, labels), [np.nan, np.nan])


def test_power_ratio2():
    epochs = np.random.randn(2000, 2, 2000)
    epochs[:1000] = epochs[:1000] * 2
    labels = [True]*1000 + [False]*1000
    npt.assert_allclose(power_ratio(epochs, labels), [4, 4], rtol=1e-2)


def test_is_good_epoch_basic(data):
    """test if detect correctly one bad channel and one flat channel among four epochs"""
    n_channels = data.shape[0]
    duration = 100
    interval = 100
    epochs = epoch(data, duration, interval, axis=1)
    epochs[0, 0, 0::] = 0  # flat channel to reject
    epochs[2, 1, 6] = 50  # spike to reject
    epochs[3, 0, 25] = 35  # too small spike to be rejected
    is_good_expected = [False, True, False, True]
    report = [[0], None, [1], None]

    channel_type_idx = dict(emg=[0, 1])
    rejection_thresholds = dict(emg=50)
    flat_thresholds = dict(emg=1e-1)

    # test only one epoch
    npt.assert_equal(_is_good_epoch(epochs[0], channel_type_idx=channel_type_idx,
                                    rejection_thresholds=rejection_thresholds,
                                    flat_thresholds=flat_thresholds), False)
    npt.assert_equal(_is_good_epoch(epochs[1], channel_type_idx=channel_type_idx,
                                    rejection_thresholds=rejection_thresholds,
                                    flat_thresholds=flat_thresholds), True)

    is_good, is_bad = is_good_epochs(epochs, channel_type_idx=channel_type_idx,
                                     rejection_thresholds=rejection_thresholds,
                                     flat_thresholds=flat_thresholds)
    npt.assert_equal(is_good, is_good_expected)
    npt.assert_equal(is_bad, report)
