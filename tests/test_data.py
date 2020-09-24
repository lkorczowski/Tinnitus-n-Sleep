import pytest
import numpy as np
from tinnsleep.data import CreateRaw, RawToEpochs_sliding, AnnotateRaw_sliding, CleanAnnotations, \
    convert_Annotations, align_labels_with_raw
import numpy.testing as npt
import mne
from collections import OrderedDict
from datetime import time
import logging
LOGGER = logging.getLogger(__name__)

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 500)

@pytest.fixture()
def dummyraw(data):
    info = mne.create_info(["Fz", "Pz"], sfreq=250.)
    return mne.io.RawArray(data, info, verbose=False)


def test_CreateRaw(data):
    ch_names = ['Fz', 'Pz']
    raw = CreateRaw(data, 1, ch_names)
    npt.assert_equal(raw.get_data(), data)


def test_CreateRaw_invalidmontage(data):
    ch_names = ['Fz', 'Pz']
    with pytest.raises(ValueError):
        raw = CreateRaw(data, 1, ch_names, montage="nice")


def test_RawToEpochs_sliding(data):
    ch_names = ['Fz', 'Pz']
    sfreq = 250
    duration = 1 * sfreq
    interval = 100
    assert RawToEpochs_sliding(CreateRaw(data, sfreq, ch_names), duration, interval, picks=None).shape == (3, 2, 250)


def test_Annotation(dummyraw):
    raw = AnnotateRaw_sliding(dummyraw, labels=[False, True, False, False, False, 2, False, False])
    expected1 = OrderedDict([('onset', .2),
                 ('duration', .2),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    expected2 = OrderedDict([('onset', 1.),
                 ('duration', .2),
                 ('description', '2'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2


def test_Annotation_withdict(dummyraw):
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = AnnotateRaw_sliding(dummyraw, labels=[0, 1, 0, 0, 0, 2, 0, 0], dict_annotations=dict_annotations)
    expected1 = OrderedDict([('onset', .2),
                 ('duration', .2),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])

    expected2 = OrderedDict([('onset', 1.),
                 ('duration', .2),
                 ('description', 'nice'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2


def test_Annotation_withinterval(dummyraw):
    interval = 50
    duration = 250
    labels = [2, 1, 3]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                   interval=interval, duration=duration)
    expected1 = OrderedDict([('onset', 0.),
                 ('duration', 1.0),
                 ('description', 'nice'),
                 ('orig_time', None)])
    expected2 = OrderedDict([('onset', .2),
                 ('duration', 1.),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])
    expected3 = OrderedDict([('onset', .4),
                 ('duration', 1.),
                 ('description', '3'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected1
    assert raw.annotations[1] == expected2
    assert raw.annotations[2] == expected3


def test_Annotation_withinterval_merge(dummyraw):
    interval = 50
    duration = 50
    labels = [1, 1, 0, 0, 1, 2, 0, 3, 3]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                   interval=interval, duration=duration, merge=True)
    expected0 = OrderedDict([('onset', .0),
                 ('duration', .4),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])
    expected1 = OrderedDict([('onset', .8),
                 ('duration', .2),
                 ('description', 'bad EPOCH'),
                 ('orig_time', None)])
    expected2 = OrderedDict([('onset', 1.0),
                 ('duration', .2),
                 ('description', 'nice'),
                 ('orig_time', None)])
    expected3 = OrderedDict([('onset', 1.40),
                 ('duration', .4),
                 ('description', '3'),
                 ('orig_time', None)])

    assert raw.annotations[0] == expected0
    assert raw.annotations[1] == expected1
    assert raw.annotations[2] == expected2
    assert raw.annotations[3] == expected3



def test_Annotation_outoflength(dummyraw):
    interval = 300
    duration = 250
    labels = [0, 1, 0]
    total_length = interval * (len(labels) - 1) + duration
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    with pytest.raises(ValueError,
                       match=f"Total length \({total_length}\) exceed length of raw \({dummyraw.__len__()}\)"):
        raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=interval, duration=duration)


def test_Annotation_invalidrange(dummyraw):
    labels = [0, 1, 0]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=0, duration=10)
    with pytest.raises(ValueError, match="Invalid range for parameters"):
        raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                       interval=10, duration=0)


def test_CleanAnnotations(dummyraw):
    raw = AnnotateRaw_sliding(dummyraw, labels=[False, True, False, False, False, True, False, False])
    raw = CleanAnnotations(raw)

    assert len(raw.annotations) == 0


def test_convert_Annotations(dummyraw):
    interval = 50
    duration = 250
    labels = [2, 1, 3]
    dict_annotations = {1: "bad EPOCH", 2: "nice"}
    raw = AnnotateRaw_sliding(dummyraw, labels=labels, dict_annotations=dict_annotations,
                   interval=interval, duration=duration)
    expected_annots = [
                    OrderedDict([('onset', 0.),
                                 ('duration', 1.0),
                                 ('description', 'nice'),
                                 ('orig_time', None)]),
                    OrderedDict([('onset', .2),
                                 ('duration', 1.),
                                 ('description', 'bad EPOCH'),
                                 ('orig_time', None)]),
                    OrderedDict([('onset', .4),
                                 ('duration', 1.),
                                 ('description', '3'),
                                 ('orig_time', None)])
                    ]

    annots = convert_Annotations(raw.annotations)

    assert annots == expected_annots


def test_align_labels_with_raw():
    timestamps = np.array(['23:30:00', '00:00:00', '00:30:00'])
    time_start = time(23, 29, 25)
    times = np.linspace(0, 1560, 1560, endpoint=False)
    npt.assert_equal(align_labels_with_raw(timestamps, time_start, times), [35, 1835, 3635])

    time_start = time(23, 30, 25)
    times = np.linspace(0, 1560, 1560, endpoint=False)
    align_labels_with_raw(timestamps, time_start, times)



def test_align_labels_with_raw_format():
    timestamps = np.array(['23:30:00.001', '00:00:00.001', '00:30:00.001'])
    time_start = time(23, 29, 25)
    times = np.linspace(0, 1560, 1560, endpoint=False)
    npt.assert_almost_equal(align_labels_with_raw(timestamps, time_start, times), [35.001, 1835.001, 3635.001])

    timestamps = np.array(['23:30:00', '00:00:00', '00:30:00'])
    time_start = time(23, 29, 25)
    times = np.linspace(0, 1560, 1560, endpoint=False)
    npt.assert_equal(align_labels_with_raw(timestamps, time_start, times), [35, 1835, 3635])