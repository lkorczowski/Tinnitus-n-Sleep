import pytest
import numpy as np
import numpy.testing as npt

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)