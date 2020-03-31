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
    raw = Annotate(dummyraw, labels=[False, True, False, False, False, 2, False, False])
    expected1 = OrderedDict([('onset', .25),
                 ('duration', .25),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    expected2 = OrderedDict([('onset', 1.25),
                 ('duration', .25),
                 ('description', '2'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2


def test_Annotation_withdict(dummyraw):
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = Annotate(dummyraw, labels=[0, 1, 0, 0, 0, 2, 0, 0], dict_annotations=dict_annotations)
    expected1 = OrderedDict([('onset', .25),
                 ('duration', .25),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    expected2 = OrderedDict([('onset', 1.25),
                 ('duration', .25),
                 ('description', 'nice'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2


def test_Annotation_withinterval(dummyraw):
    interval = 50
    duration = 200
    labels = [2, 1, 3]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = Annotate(dummyraw, labels=labels, dict_annotations=dict_annotations,
                   interval=interval, duration=duration)
    expected1 = OrderedDict([('onset', 0.),
                 ('duration', 1.0),
                 ('description', 'nice'),
                 ('orig_time', None)])
    expected2 = OrderedDict([('onset', .25),
                 ('duration', 1.),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])
    expected3 = OrderedDict([('onset', .5),
                 ('duration', 1.),
                 ('description', '3'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2
    assert raw.annotations[2] == expected3


def test_Annotation_outoflength(dummyraw):
    interval = 300
    duration = 200
    labels = [0, 1, 0]
    total_length = interval * (len(labels) - 1) + duration
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    with pytest.raises(ValueError,
                       match=f"Total length \({total_length}\) exceed length of raw \({dummyraw.__len__()}\)"):
        raw = Annotate(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=interval, duration=duration)


def test_Annotation_invalidrange(dummyraw):
    labels = [0, 1, 0]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        raw = Annotate(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=0, duration=10)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        raw = Annotate(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=10, duration=0)


def test_CleanAnnotations(dummyraw):
    raw = Annotate(dummyraw, labels=[False, True, False, False, False, True, False, False])
    raw = CleanAnnotations(raw)

    assert len(raw.annotations) == 0
