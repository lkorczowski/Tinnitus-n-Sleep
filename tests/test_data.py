import pytest
import numpy as np
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
import numpy.testing as npt

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)

def test_CreateRaw(data):
    ch_names = ['Fz', 'Pz']
    raw = CreateRaw(data, ch_names)
    npt.assert_equal(raw.get_data(), data)

def test_CreateRaw_invalidmontage(data):
    ch_names = ['Fz', 'Pz']
    with pytest.raises(ValueError, match="Could not find the montage. Please provide the full path."):
        raw = CreateRaw(data, ch_names, montage="nice")

def test_RawToEpochs_sliding(data):
    ch_names = ['Fz', 'Pz']
    duration = 1-1/200
    interval = 0.5
    assert RawToEpochs_sliding(CreateRaw(data, ch_names), duration, interval, picks=None).get_data().shape \
           == (3, 2, 200)
