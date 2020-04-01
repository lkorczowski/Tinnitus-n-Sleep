import pytest
import numpy as np
from tinnsleep.visualization import plotTimeSeries
import matplotlib.pyplot as plt


def test_plotTimeSeries():
    np.random.seed(42)
    data = np.random.randn(400, 2)
    plotTimeSeries(data)
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