import pytest
import numpy as np
from tinnsleep.visualization import plotTimeSeries, assert_ax_equals_data, assert_x_labels_correct
import matplotlib.pyplot as plt
import numpy.testing as npt


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
        assert_x_labels_correct(data, ["0" for k in range(data.shape[1])])

    np.random.seed(10)
    data = np.random.randn(400, 2)

    # check if y-label position are incorrect
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_x_labels_correct(data, ["0" for k in range(data.shape[1])])

    # check if data values are incorrect values
    with pytest.raises(AssertionError, match=r"Items are not equal to 7 significant digits:"):
        # check if correct values
        assert_ax_equals_data(data, ax, sfreq=sfreq)


def test_plotTimeSeries_noparams():
    """Complete test suite for plotTimeSeries
    """
    plt.close("all")
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data)

    # check if label position and values are correct
    assert_x_labels_correct(data, [str(k) for k in range(data.shape[1])])

    # check if correct values
    assert_ax_equals_data(data, ax, sfreq=sfreq)


def test_plotTimeSeries_superimpose():
    """Test if we can superimpose several timeseries
    """
    plt.close("all")
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, color="red")

    np.random.seed(42)
    data = np.random.randn(400, 4)
    fig, ax = plotTimeSeries(data, ax=ax, color="black", linestyle="--")
    plt.legend()
    plt.show()


def test_plotTimeSeries_chnames_propagation():
    """test if ch_names propagate to all channels
    """
    plt.close("all")
    sfreq = 1
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, ch_names="EMG")
    # check if label position and values are correct
    assert_x_labels_correct(data, ['EMG' for k in range(data.shape[1])])


def test_plotTimeSeries_subplots():
    """Test if two axes can be managed
    """
    plt.close("all")
    sfreq=100
    np.random.seed(42)
    data = np.random.randn(400, 2)
    ax = plt.subplot(2, 1, 2)
    ch_names=["You underestimate", "my power"]
    fig, ax = plotTimeSeries(data, ax=ax, color="r", marker=".", linestyle='dashed',
                             linewidth=2, markersize=0.5, ch_names=ch_names, sfreq=sfreq)
    plt.title("lava platform")
    assert_ax_equals_data(data, ax, sfreq=sfreq)
    assert_x_labels_correct(data, ch_names)

    sfreq=200
    np.random.seed(42)
    data = np.random.randn(400, 4)
    ax = plt.subplot(2, 1, 1)
    ch_names = ["Its over", "I have the", "high", "ground"]
    fig, ax = plotTimeSeries(data, ax=ax, color="b", marker="*", linestyle='-',
                             linewidth=2, markersize=0.5, ch_names=ch_names, sfreq=sfreq)
    plt.title("higher ground")
    assert_ax_equals_data(data, ax, sfreq=sfreq)
    assert_x_labels_correct(data, ch_names)


def test_plotTimeSeries_1dim():
    np.random.seed(42)
    data = np.random.randn(10)
    plotTimeSeries(data)


def test_plotTimeSeries_incorrectdim():
    np.random.seed(42)
    data = np.random.randn(1, 2, 3)
    with pytest.raises(ValueError, match="data should be two-dimensional"):
        plotTimeSeries(data)


def test_plotTimeSeries_incorrect_parameters():
    np.random.seed(42)
    data = np.random.randn(400, 4)

    with pytest.raises(ValueError, match="\`ch_names\` must be a list or an iterable of shape \(n_dimension,\) or None"):
        plotTimeSeries(data, ch_names=True)

    with pytest.raises(ValueError, match='ch_names should be same length as the number of channels of data'):
        plotTimeSeries(data, ch_names=[1, 2])

    with pytest.raises(ValueError, match="\`ax\` must be a matplotlib Axes instance or None"):
        plotTimeSeries(data, ax=True)

