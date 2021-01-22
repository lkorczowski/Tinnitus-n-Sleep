import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms, is_good_epochs, _is_good_epoch, power_ratio, align_epochs_latency
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


def test_align_epochs_latency_dirac():
    m_shape = (2, 25)
    list_epochs = [np.zeros(m_shape), np.zeros(m_shape)]
    list_epochs[0][1, 10] = 1
    list_epochs[1][1, 20] = 1
    npt.assert_equal(list_epochs[1][:, 20], list_epochs[0][:, 10])
    list_argmax = []
    for i, x in enumerate(list_epochs):
        list_argmax.append(np.argmax(x))

    list_shifts = min(list_argmax) - list_argmax
    list_shifts = [5, -5]  # using both right hand and left hand padding
    list_epochs = align_epochs_latency(list_epochs, list_shifts)

    npt.assert_equal(list_epochs[1], list_epochs[0])


def test_align_epochs_latency_dirac_different_length():
    m_shape = (2, 25)
    m_shape2 = (2, 30)
    list_epochs = [np.zeros(m_shape), np.zeros(m_shape2)]
    list_epochs[0][1, 10] = 1
    list_epochs[1][1, 20] = 1
    npt.assert_equal(list_epochs[1][:, 20], list_epochs[0][:, 10])
    list_argmax = []
    for i, x in enumerate(list_epochs):
        list_argmax.append(np.argmax(x))

    list_shifts = min(list_argmax) - list_argmax
    list_shifts = [5, -5]  # using both right hand and left hand padding
    list_epochs = align_epochs_latency(list_epochs, list_shifts)

    npt.assert_equal(list_epochs[1][:, :m_shape[1]], list_epochs[0])


def test_align_epochs_latency_dirac_multiple_left():
    # calibrate all dirac of random latency to the earliest dirac
    n_epochs = 10000
    n_channels = 10
    list_epochs = []
    list_argmax = []
    min_samples = 1000
    max_samples = 0
    for i in range(n_epochs):
        n_samples = np.random.randint(25, 1000)


        if n_samples<min_samples:
            min_samples = n_samples
        if n_samples > max_samples:
            max_samples = n_samples

        list_argmax.append(np.random.randint(10, n_samples))
        list_epochs.append(np.zeros((n_channels, n_samples)))
        list_epochs[-1][:, list_argmax[-1]] = 1

    list_shifts = min(list_argmax) - np.array(list_argmax)

    list_epochs_aligned = align_epochs_latency(list_epochs, list_shifts)

    for i in range(len(list_epochs_aligned)):
        npt.assert_equal(list_epochs_aligned[i][:, :min_samples], list_epochs_aligned[0][:, :min_samples])


def test_align_epochs_latency_dirac_multiple_right():
    # calibrate all dirac of random latency to the latest dirac
    n_epochs = 10000
    n_channels = 10
    list_epochs = []
    list_argmax = []
    min_samples = 1000
    max_samples = 0
    for i in range(n_epochs):
        n_samples = np.random.randint(25, 1000)


        if n_samples<min_samples:
            min_samples = n_samples
        if n_samples > max_samples:
            max_samples = n_samples

        list_argmax.append(np.random.randint(0, n_samples))
        list_epochs.append(np.zeros((n_channels, n_samples)))
        list_epochs[-1][:, list_argmax[-1]] = 1


    # calibrate all dirac of random latency to the latest dirac

    list_shifts2 = max(list_argmax) - np.array(list_argmax)

    list_epochs_aligned = align_epochs_latency(list_epochs, list_shifts2)

    for i in range(len(list_epochs_aligned)):
        npt.assert_equal(list_epochs[i][:, :max_samples], list_epochs[0][:,:max_samples])

