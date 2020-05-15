import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.utils import epoch, compute_nb_epochs, merge_labels_list, fuse_with_classif_result


def test_compute_nb_epochs():
    assert compute_nb_epochs(10, 5, 5) == 2



def test_compute_nb_epochs_invalid():
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        compute_nb_epochs(10, 0, 5)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        compute_nb_epochs(10, 1, 0)


def test_epoch_unit1():
    np.random.seed(seed=42)
    N = 1000               # signal length
    T = 100                # window length
    I = 101                # interval
    Ne = 8                 # electrodes
    window_length = 1      # in seconds
    window_overlap = 0     # in seconds
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)


def test_epoch_unit2():
    np.random.seed(seed=42)
    N = 100               # signal length
    T = 100                # window length
    I = 101                # interval
    Ne = 8                 # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X, T, I)
    assert epochs.shape == (K, Ne, T)


def test_epoch_unit_with_axis1():
    np.random.seed(seed=42)
    N = 1000               # signal length
    T = 100                # window length
    I = 10                 # interval
    Ne = 8                 # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X.T, T, I, axis=0)
    assert epochs.shape == (K, T, Ne)


def test_epoch_unit_with_axis2():
    epochs_target = np.array([[[1,  2,  3,  4]],
                              [[4,  5,  6,  7]],
                              [[7,  8,  9, 10]]])
    X = np.expand_dims(np.arange(1, 11), axis=0)
    T = 4                 # window length
    I = 3                  # interval
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


def test_fuse_with_classif_result():
    check_imp = [[False, False], [False, True], [True, True], [True, False], [True, True], [True, False]]
    classif=np.asanyarray([1, 2, 3, 4])
    classif = fuse_with_classif_result(check_imp, classif)
    npt.assert_equal(classif, [1, 2, False, 3, False, 4])
