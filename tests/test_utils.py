import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.utils import epoch, compute_nb_epochs

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

def test_epoch_unit_with_axis():
    np.random.seed(seed=42)
    N = 1000               # signal length
    T = 100                # window length
    I = 10                 # interval
    Ne = 8                 # electrodes
    X = np.random.randn(Ne, N)
    K = compute_nb_epochs(N, T, I)
    epochs = epoch(X.T, T, I, axis=0)
    assert epochs.shape == (K, T, Ne)

def test_epoch_unit_with_axis():
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