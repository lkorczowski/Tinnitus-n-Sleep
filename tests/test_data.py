import pytest
import numpy as np
from tinnsleep.data import CreateRaw, RawToEpochs_sliding, Annotate, CleanAnnotations
import numpy.testing as npt
import mne
from collections import OrderedDict

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)

@pytest.fixture()
def dummyraw(data):
    info = mne.create_info(["Fz", "Pz"], sfreq=200.)
    return mne.io.RawArray(data, info, verbose=False)


def test_CreateRaw(data):
    ch_names = ['Fz', 'Pz']
    raw = CreateRaw(data, ch_names)
    npt.assert_equal(raw.get_data(), data)


def test_CreateRaw_invalidmontage(data):
    ch_names = ['Fz', 'Pz']
    with pytest.raises(ValueError):
        raw = CreateRaw(data, ch_names, montage="nice")


def test_RawToEpochs_sliding(data):
    ch_names = ['Fz', 'Pz']
    duration = 200
    interval = 100
    assert RawToEpochs_sliding(CreateRaw(data, ch_names), duration, interval, picks=None).shape == (3, 2, 200)


def test_Annotation(dummyraw):
    raw = Annotate(dummyraw, labels=[False, True, False, False, False, True, False, False])
    expected1 = OrderedDict([('onset', .25),
                 ('duration', .25),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    expected2 = OrderedDict([('onset', 1.25),
                 ('duration', .25),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2


def test_CleanAnnotations(dummyraw):
    raw = Annotate(dummyraw, labels=[False, True, False, False, False, True, False, False])
    raw = CleanAnnotations(raw)

    assert len(raw.annotations) == 0
