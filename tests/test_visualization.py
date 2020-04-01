import pytest
import numpy as np
from tinnsleep.visualization import plotTimeSeries

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)


def test_plotTimeSeries(data):
    plotTimeSeries(data)
