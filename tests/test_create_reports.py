import pytest
import numpy as np
from tinnsleep.data import CreateRaw
from tinnsleep.create_reports import  preprocess, reporting
import numpy.testing as npt
from tinnsleep.utils import epoch


def test_preprocess():
    np.random.seed(42)
    data = np.random.randn(4, 400)

    data[0] = 1e-7 * data[0]
    data[1] = 1e-7 * data[1]
    data[1][100:200] += 100
    data[2][:100] += 100
    data[3][:100] += 100



    ch_names = ['1', '2', 'IMP_1', 'IMP_2']
    raw = CreateRaw(data, ch_names, ch_types=["emg"])
    picks_chan = ['1', '2']
    picks_imp = ['IMP_1', 'IMP_2']
    duration = 50
    interval = 50
    print(raw.info["ch_names"])
    params = dict(ch_names=['1', '2'],
                  rejection_thresholds=dict(emg=1e-04),  # two order of magnitude higher q0.01
                  flat_thresholds=dict(emg=1e-09),  # one order of magnitude lower median
                  channel_type_idx=dict(emg=[0, 1]),
                  full_report=True
                  )

    epochs, valid_labels = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50, get_log=False)
    npt.assert_equal(len(epochs), 8)
    npt.assert_equal(valid_labels, [False, False, False, False, True, True, True, True])
    epochs, valid_labels, log = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50, get_log=True)
    npt.assert_equal(log, {'suppressed_imp_THR': 2, 'suppressed_amp_THR': 2, 'suppressed_overall': 4})


def test_reporting():
    np.random.seed(42)
    data = np.random.randn(2, 800)

    data[0] = 1e-7 * data[0]
    data[2] = 1e-7 * data[2]
    data[0][100:150] += 100
    data[1][100:150] += 100
    data[0][200:250] += 100
    data[1][200:250] += 100
    data[0][300:350] += 100
    data[1][300:350] += 100
    data[0][400:450] += 100
    data[1][400:450] += 100

    duration = 50
    interval = 50

    epochs = epoch(data, duration, interval, axis=-1)
    THR_classif = [[0, 2], [0, 3]]
    valid_labels = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    report = reporting(epochs, valid_labels, THR_classif, log={})
    npt.assert_equal(report["labels"][0], [False, False, False, False, True, False, True, False, True, False, False,
                                           False, False, False, False, False])
    npt.assert_equal(report["reports"][0]['Total number of burst'], 3)
    npt.assert_equal(report["reports"][0]['Mean duration of phasic episode'], 1.25)