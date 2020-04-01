import pytest
import numpy as np
from tinnsleep.visualization import plotTimeSeries
import matplotlib.pyplot as plt


def test_plotTimeSeries_superimpose():
    """Test if two axes can be managed
    """
    np.random.seed(42)
    data = np.random.randn(400, 2)
    fig, ax = plotTimeSeries(data, color="red")

    np.random.seed(42)
    data = np.random.randn(400, 4)
    fig, ax = plotTimeSeries(data, ax=ax, color="black", linestyle="--")
    plt.show()


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