import pytest
import numpy as np
from tinnsleep.check_impedance import check_RMS, create_annotation_sliding, Impedance_thresholding_sliding, create_annotation_mne
import numpy.testing as npt


def test_Impedance_thresholding_sliding():
    np.random.seed(42)
    data = np.random.randn(2, 400)
    for i in range(200):
        data[0][i] += 100
    for i in range(100):
        data[1][i] += 100
    ch_names = ['1_imp', '2_imp']
    duration = 50
    interval = 50
    THR = 20.0
    check_imp = Impedance_thresholding_sliding(data, duration, interval, THR)
    npt.assert_equal(check_imp, [[True, True], [True, True], [True, False], [True, False], [False, False], [False, False],
                                 [False, False], [False, False]])

def test_check_RMS():
    check_imp = [[False, False], [False, True], [True, True], [True, False]]
    RMS=[]
    with pytest.raises(ValueError, match="Inputs shapes don't match"):
        check_RMS(RMS, check_imp)
    RMS = [[1, 1], [1, 12], [12, 12], [1, 1]]
    RMS = check_RMS(RMS, check_imp)
    npt.assert_equal(RMS, [[1, 1], [1, 1], [1, 1]])


def test_bad_epochs_annotations():
    check_imp = [[False, False], [False, True], [True, True], [True, False]]
    duration = 50
    interval = 50
    anno = create_annotation_sliding(check_imp, duration, interval)
    npt.assert_equal(anno, [{'onset': 100.0, 'duration': 50.0, 'description': "1", 'orig_time': 0.0}])

def test_annotation_mne():
    check_imp = [[False, False], [False, True], [True, True], [True, False]]
    anno = create_annotation_mne(check_imp)
    npt.assert_equal(anno, [False, False, True, False])
