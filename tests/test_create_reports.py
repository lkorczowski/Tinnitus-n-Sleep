import pytest
import numpy as np
from tinnsleep.data import CreateRaw
from tinnsleep.create_reports import setting_channels, preprocess, reporting
import numpy.testing as npt
from tinnsleep.utils import epoch


def test_setting_channels():
    np.random.seed(42)
    data = np.random.randn(4, 200)
    ch_names = ['1', 'IMP_1', '2', 'IMP_2']
    raw = CreateRaw(data, ch_names)
    ind_picks_chan = [0, 2]
    ind_picks_imp = [1, 3]
    pick_chan, pick_imp = setting_channels(raw, ind_picks_chan, ind_picks_imp)

    npt.assert_equal(pick_chan, ["1", "2"])
    npt.assert_equal(pick_imp, ["IMP_1", "IMP_2"])

def test_preprocess():
    np.random.seed(42)
    data = np.random.randn(4, 400)
    for i in range(400):
        data[0][i] = data[0][i] * 0.00001
        data[2][i] = data[2][i] * 0.00001

    for i in range(100):
        data[1][i] += 100
    for i in range(100):
        data[3][i] += 100
    for i in range(100):
        data[2][i+100] += 100
    ch_names = ['1', 'IMP_1', '2', 'IMP_2']
    raw = CreateRaw(data, ch_names, ch_types=["emg"])
    picks_chan = ['1', '2']
    picks_imp = ['IMP_1', 'IMP_2']
    duration = 50
    interval = 50

    epochs, valid_labels = preprocess(raw, picks_chan, picks_imp, duration, interval, THR_imp=50, get_log=False)

    npt.assert_equal(len(epochs), 8)
    npt.assert_equal(valid_labels, [False, False, False, False, False, False, False, False])

    epochs, valid_labels, log = preprocess(raw, picks_chan, picks_imp, duration, interval, THR_imp=50, get_log=True)
    npt.assert_equal(log, {'suppressed_imp_THR': 2, 'suppressed_amp_THR': 8, 'suppressed_overall': 8})


def test_reporting():
    np.random.seed(42)
    data = np.random.randn(2, 400)
    for i in range(400):
        data[0][i] = data[0][i] * 0.00001
        data[1][i] = data[1][i] * 0.00001

    for i in range(150):
        data[0][i + 100] += 100


    duration = 50
    interval = 50

    epochs = epoch(data, duration, interval, axis=-1)
    THR_classif = [[0, 2], [0, 3]]
    valid_labels = [True, True, False, True, True, True, True, True]
    print(reporting(epochs, valid_labels, THR_classif, log={}))