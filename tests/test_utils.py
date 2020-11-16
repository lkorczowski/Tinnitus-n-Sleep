import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.utils import epoch, compute_nb_epochs, merge_labels_list, \
    fuse_with_classif_result, crop_to_proportional_length, resample_labels, label_report, \
    merge_label_and_events, print_dict, round_time, labels_1s_extension
from scipy.interpolate import interp1d
import datetime


def test_compute_nb_epochs():
    assert compute_nb_epochs(10, 5, 5) == 2


def test_compute_nb_epochs_invalid():
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        compute_nb_epochs(10, 0, 5)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        compute_nb_epochs(10, 1, 0)


def test_epoch_unit1():
    np.random.seed(seed=42)
    N = 1000  # signal length
    T = 100  # window length
    I = 101  # interval
    Ne = 8  # electrodes
    window_length = 1  # in seconds
    window_overlap = 0  # in seconds
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)


def test_epoch_unit2():
    np.random.seed(seed=42)
    N = 100  # signal length
    T = 100  # window length
    I = 101  # interval
    Ne = 8  # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)


def test_epoch_unit_with_axis1():
    np.random.seed(seed=42)
    N = 1000  # signal length
    T = 100  # window length
    I = 10  # interval
    Ne = 8  # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X.T, T, I, axis=0)
    assert epochs.shape == (K, T, Ne)


def test_epoch_unit_with_axis2():
    epochs_target = np.array([[[1, 2, 3, 4]],
                              [[4, 5, 6, 7]],
                              [[7, 8, 9, 10]]])
    X = np.expand_dims(np.arange(1, 11), axis=0)
    T = 4  # window length
    I = 3  # interval
    epochs = epoch(X, T, I, axis=1)
    npt.assert_array_equal(epochs, epochs_target)


def test_epoch_fail_size():
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        np.random.seed(seed=42)
        N = 1000  # signal length
        T = 100  # window length
        I = 0  # interval
        Ne = 8  # electrodes
        X = np.random.randn(Ne, N)
        epochs = epoch(X.T, T, I, axis=0)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        np.random.seed(seed=42)
        N = 1000  # signal length
        T = 0  # window length
        I = 1  # interval
        Ne = 8  # electrodes
        X = np.random.randn(Ne, N)
        epochs = epoch(X.T, T, I, axis=0)


def test_merge_labels_list_ident():
    # Unchanging a list to the same number of elements
    v_lab = merge_labels_list([[True, False, True, False, True]], 5)
    npt.assert_equal(v_lab, [True, False, True, False, True])

    # Unchanging two identical list into one:
    v_lab = merge_labels_list([[True, False, True, False, True], [True, False, True, False, True]], 5)
    npt.assert_equal(v_lab, [True, False, True, False, True])


def test_merge_labels_list_proportional_upsampling():
    # dealing with 2 coherent arrays of len(l) and 2*len(l) and getting output in 2*len(l):
    v_lab = merge_labels_list([[True, False], [True, True, False, False]], 4)
    npt.assert_equal(v_lab, [True, True, False, False])

    # dealing with classic situation:
    v_lab = merge_labels_list([[True, False], [True, True, True, True]], 4)
    npt.assert_equal(v_lab, [True, True, False, False])

    # dealing with classic situation 2:
    v_lab = merge_labels_list([[True, True], [True, True, False, True]], 4)
    npt.assert_equal(v_lab, [True, True, False, True])


def test_merge_labels_list_proportional_downsampling():
    # dealing with 2 coherent arrays of len(l) and 2*len(l) and getting output in len(l):
    v_lab = merge_labels_list([[True, False], [True, True, False, False]], 2)
    npt.assert_equal(v_lab, [True, False])

    # dealing with tricky case 1:
    v_lab = merge_labels_list([[True, True], [True, True, True, False]], 2)
    npt.assert_equal(v_lab, [True, False])

    # dealing with tricky case 2:
    v_lab = merge_labels_list([[True, True, False, True], [True, True]], 2)
    npt.assert_equal(v_lab, [True, False])


def test_merge_labels_list_non_proportional():
    # downsampling interpolation
    v_lab = merge_labels_list([[True, True, False, False, False]], 2)
    npt.assert_equal(v_lab, [True, False])
    # upsampling interpolation
    v_lab = merge_labels_list([[True, False]], 5)
    npt.assert_equal(v_lab, [True, True, True, False, False])

    # downsampling interpolation
    v_lab = merge_labels_list([[True, True, True, True, True, False, False, False, False]], 2, kind='nearest')
    npt.assert_equal(v_lab, [True, False])
    # upsampling interpolation
    v_lab = merge_labels_list([[True, False]], 9)
    npt.assert_equal(v_lab, [True, True, True, True, True, False, False, False, False])


def test_resample_labels():
    # downsampling interpolation
    labels = ["A", "B", "C", "D", "E"]
    n_epochs = len(labels)
    n_epoch_new = 8
    x = [-1, 1, 2, 3, 4]
    xnew = np.linspace(0, n_epochs, n_epoch_new, endpoint=False)
    f = interp1d(x, range(len(labels)), kind='previous', fill_value=(0, len(x) - 1), bounds_error=False)
    y_idx = f(xnew)
    labels_new = [labels[int(i)] for i in y_idx]

    # ad-hoc regression tests
    npt.assert_equal(resample_labels(labels, xnew), labels_new)
    npt.assert_equal(resample_labels(labels, n_epoch_new), labels_new)
    npt.assert_equal(resample_labels(labels, xnew, x), labels_new)

    # simple regression tests
    npt.assert_equal(resample_labels(labels, 2), ["A", "C"])
    npt.assert_equal(resample_labels(labels, [-10, 1.5]), ["A", "B"])
    npt.assert_equal(resample_labels(labels, [-10, 1000]), ["A", "E"])

    # test non-monotonical arrangement
    npt.assert_equal(resample_labels(labels, [-2, -10, 1000], [-20, -5, 20, 1500, 10000]), ["B", "A", "C"])
    npt.assert_equal(resample_labels(labels, [-2, -10, 1000], [-20, -5, 20, 1500, 10000], kind='nearest'),
                     ["B", "B", "D"])

    # error with wrong parameter
    with pytest.raises(ValueError, match="Number of labels is 5, number of associated timestamps is 4"):
        resample_labels(labels, 2, [0, 1, 2, 3])


def test_fuse_with_classif_result():
    check_imp = [[False, False], [False, True], [True, True], [True, False], [True, True], [True, False]]
    classif = np.asanyarray([1, 2, 3, 4])
    classif = fuse_with_classif_result(check_imp, classif)
    npt.assert_equal(classif, [1, 2, False, 3, False, 4])


def test_crop_to_proportional_length():
    epochs = np.ones((5, 2, 2))
    valid_labels = [[True] * 5, [True, False]]
    epochs, valid_labels = crop_to_proportional_length(epochs, valid_labels)
    npt.assert_equal(epochs.shape, (4, 2, 2))
    npt.assert_equal(valid_labels, [True, True, False, False])

    epochs = np.ones((8, 2, 2))
    valid_labels = [[True] * 8, [True, True, False]]
    epochs, valid_labels = crop_to_proportional_length(epochs, valid_labels)
    npt.assert_equal(epochs.shape, (6, 2, 2))
    npt.assert_equal(valid_labels, [True, True, True, True, False, False])


def test_crop_to_proportional_length_fails():
    """ non proportional epochs with number of valid_labels"""
    with pytest.raises(AssertionError):
        epochs = np.ones((3, 2, 2))
        valid_labels = [[True] * 5, [True, False]]
        crop_to_proportional_length(epochs, valid_labels)


def test_generate_label_report_basic():
    sleep_labels = np.array(["awake", "N1", "N1", "N2", "invalid", "N3", "NREM", "awake"])
    report = label_report(sleep_labels)
    report_expected = {'N1 count': 2,
                         'N1 ratio': 0.25,
                         'N2 count': 1,
                         'N2 ratio': 0.125,
                         'N3 count': 1,
                         'N3 ratio': 0.125,
                         'NREM count': 1,
                         'NREM ratio': 0.125,
                         'awake count': 2,
                         'awake ratio': 0.25,
                         'invalid count': 1,
                         'invalid ratio': 0.125}
    npt.assert_equal(report, report_expected)


def test_generate_sleep_report_with_episodes():
    time_interval = 1.0
    event_start = [0.5, 5.3, 5.8]
    sleep_labels = np.array(["awake", "N1", "N1", "N2", "invalid", "N3", "REM", "awake"])
    npt.assert_equal(merge_label_and_events(event_start, sleep_labels, time_interval), ["awake", "N3", "N3"])


def test_print_dict():
    a = {'a': 1}
    print_dict(a)


def test_round_time():
    # hour rounding of hte current time
    now = datetime.datetime.now()
    npt.assert_equal(round_time(round_to=3600),
                     datetime.datetime(now.year, now.month, \
                                       now.day, now.hour + (now.minute > 29))
                     )

    # minute rounding
    dt = datetime.datetime(1900, 1, 1, 23, 59, 31)
    npt.assert_equal(round_time(dt=dt, round_to=60), datetime.datetime(1900, 1, 2))

    # hour rounding
    dt = datetime.datetime(1900, 1, 1, 23, 31, 31)
    npt.assert_equal(round_time(dt=dt, round_to=60 * 60), datetime.datetime(1900, 1, 2))

    # day rounding
    dt = datetime.datetime(1900, 1, 1, 13, 31, 31)
    npt.assert_equal(round_time(dt=dt, round_to=60 * 60 * 24), datetime.datetime(1900, 1, 2))

    # 30-sec rounding
    dt = datetime.datetime(1900, 1, 1, 23, 59, 44)
    npt.assert_equal(round_time(dt=dt, round_to=30), datetime.datetime(1900, 1, 1, 23, 59, 30))

    # 500 ms rounding (not actually working now)
    with pytest.raises(ValueError):
        dt = datetime.datetime(1900, 1, 1, 23, 59, 58, 990)
        round_time(dt=dt, round_to=0.5)

def test_extending_episodes_right():
    l=[0]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [0])
    l=[1]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [1])
    l=[0,0]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [0,0])
    l=[0,1]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [0,1])
    l=[1,0]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [1,1])
    l=[1,1]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [1,1])
    l=[1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1]
    n_l = extending_episodes_right(l)
    npt.assert_equal(n_l, [1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1])

def test_extending_episodes_left():
    l=[0]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [0])
    l=[1]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [1])
    l=[0,0]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [0,0])
    l=[0,1]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [1,1])
    l=[1,0]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [1,0])
    l=[1,1]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [1,1])
    l=[1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1]
    n_l = extending_episodes_left(l)
    npt.assert_equal(n_l, [1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1])

def test_labels_1s_extension():
    l=[0]
    npt.assert_equal(labels_1s_extension(l,1,1), [0])
    l=[1]
    npt.assert_equal(labels_1s_extension(l, 1, 1), [1])
    l=[0,0]
    npt.assert_equal(labels_1s_extension(l, 1, 1), [0,0])
    npt.assert_equal(labels_1s_extension(l, 1, 0), [0,0])
    npt.assert_equal(labels_1s_extension(l, 0, 1), [0,0])
    l=[0,1]
    npt.assert_equal(labels_1s_extension(l, 1, 1), [1,1])
    l=[1,0]
    npt.assert_equal(labels_1s_extension(l, 1, 1), [1,1])
    l=[1,1]
    npt.assert_equal(labels_1s_extension(l, 1, 1), [1,1])
    l=[1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1]
    npt.assert_equal(labels_1s_extension(l, 0, 0), [1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1])
    npt.assert_equal(labels_1s_extension(l, 1, 0), [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    npt.assert_equal(labels_1s_extension(l, 0, 1), [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    npt.assert_equal(labels_1s_extension(l, 1, 1), [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    npt.assert_equal(labels_1s_extension(l, 1, 2), [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
