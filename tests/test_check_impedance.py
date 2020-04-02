import pytest
import numpy as np
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.check_impedance import check_RMS, create_bad_epochs_annotation, Impedance_thresholding
import numpy.testing as npt

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(2, 400)

def test_Impedance_thresholding():
    np.random.seed(42)
    data = np.random.randn(2, 400)
    for i in range(200):
        data[0][i] += 100
        data[1][i] += 100
    ch_names = ['1_imp', '2_imp']
    duration = 50
    interval = 50
    THR = 20.0
    check_imp = Impedance_thresholding(data, ch_names, duration, interval, THR, ch_types=['emg'])
    npt.assert_equal(check_imp, [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]])

def test_check_RMS():
    check_imp = [[1, 1], [1, 0], [0, 1], [0, 0]]
    RMS=[]
    with pytest.raises(ValueError, match="Inputs shapes don't match"):
        check_RMS(RMS, check_imp)
    RMS = [[1, 1], [1, 12], [12, 1], [12, 12]]
    RMS = check_RMS(RMS, check_imp)
    print(RMS)
    npt.assert_equal(RMS, [[1, 1], [1, 1], [1, 1]])

def test_bad_epochs_annotations():
    check_imp = [[0, 1], [1, 0], [1, 1], [0, 0]]
    duration = 50
    interval = 50
    anno = create_bad_epochs_annotation(check_imp, duration, interval)
    npt.assert_equal(anno, [{'onset': 100.0, 'duration': 50.0, 'description': "1", 'orig_time': 0.0}])
