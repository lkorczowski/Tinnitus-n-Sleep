import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.data import RawToEpochs_sliding, CreateRaw
from tinnsleep.visualization import create_visual
from tinnsleep.utils import compute_nb_epochs

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)

def test_visualization(data):
    ch_names = ['Fz', 'Pz']
    duration = 100  # in samples
    interval = 50  # in samples
    raw = CreateRaw(data, ch_names)
    leny = compute_nb_epochs(400, duration, interval)
    b_detect = [False] * leny
    b_detect[1] = True
    create_visual(raw, b_detect, duration, interval)