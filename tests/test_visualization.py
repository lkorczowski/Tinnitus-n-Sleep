import pytest
import numpy as np
from tinnsleep.visualization import plotTimeSeries
import matplotlib.pyplot as plt


def test_plotTimeSeries_superimpose():
    """Test if we can superimpose several timeseries
    """
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, color="red")

    np.random.seed(42)
    data = np.random.randn(400, 4)
    fig, ax = plotTimeSeries(data, ax=ax, color="black", linestyle="--")
    plt.show()


def test_plotTimeSeries_chnames_propagation():
    """test if ch_names propagate to all channels
    """
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, ch_names="EMG")


def test_plotTimeSeries_subplots():
    """Test if two axes can be managed
    """
    np.random.seed(42)
    data = np.random.randn(400, 2)
    ax = plt.subplot(2, 1, 2)
    fig, ax = plotTimeSeries(data, ax=ax, color="r", marker=".", linestyle='dashed',
                             linewidth=2, markersize=0.5, ch_names=["You underestimate", "my power"])
    plt.title("lava platform")

    np.random.seed(42)
    data = np.random.randn(400, 4)
    ax = plt.subplot(2, 1, 1)
    fig, ax = plotTimeSeries(data, ax=ax, color="b", marker="*", linestyle='-',
                             linewidth=2, markersize=0.5, ch_names=["Its over", "I have the", "high", "ground"])
    plt.title("higher ground")

    plt.show()


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

