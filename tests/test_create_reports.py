import pytest
import numpy as np
from tinnsleep.data import CreateRaw
from tinnsleep.create_reports import preprocess, reporting, merge_labels_list, combine_brux_MEMA
import numpy.testing as npt
from tinnsleep.utils import epoch
from tinnsleep.scoring import generate_bruxism_report, generate_MEMA_report


def test_preprocess():
    np.random.seed(42)
    data = np.random.randn(4, 400)

    data[0] = 1e-6 * data[0]
    data[1] = 1e-6 * data[1]
    data[1][110:120] = 1000 * data[1][110:120]
    data[2][:100] += 100
    data[3][:100] += 100

    sfreq = 250
    ch_names = ['1', '2', 'IMP_1', 'IMP_2']
    raw = CreateRaw(data, sfreq, ch_names, ch_types="emg")
    picks_chan = ['1', '2']
    picks_imp = ['IMP_1', 'IMP_2']
    duration = 50
    interval = 50
    params = dict(ch_names=['1', '2'],
                  rejection_thresholds=dict(emg=1e-04),  # two order of magnitude higher q0.01
                  flat_thresholds=dict(emg=1e-09),  # one order of magnitude lower median
                  channel_type_idx=dict(emg=[0, 1]),
                  full_report=True
                  )

    epochs, valid_labels = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50, get_log=False)
    npt.assert_equal(len(epochs), 8)

    epochs, valid_labels, log = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50,
                                           get_log=True, filter=None)
    npt.assert_equal(log, {'suppressed_imp_THR': 2, 'suppressed_amp_THR': 1, 'suppressed_overall': 3,
                           'total_nb_epochs': 8})
    npt.assert_equal(valid_labels, [False, False, False, True, True, True, True, True])

    with pytest.raises(ValueError, match=r'`filter` should be default, a dict of parameters to pass to raw.filter, '
                                         r'or None'):
        epochs, valid_labels, log = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50,
                                               get_log=True, filter="nice")

    filtering = {"l_freq" : 21., "h_freq" : 99., "n_jobs" : 4,
                         "fir_design" : 'firwin', "filter_length" : 'auto', "phase" :'zero-double',
                         "picks" : picks_chan}
    epochs, valid_labels, log = preprocess(raw, picks_chan, picks_imp, duration, interval, params, THR_imp=50,
                                           get_log=True, filter=filtering)

    npt.assert_equal(log, {'suppressed_imp_THR': 2, 'suppressed_amp_THR': 1, 'suppressed_overall': 3,
                           'total_nb_epochs': 8})
    npt.assert_equal(valid_labels, [False, False, False, True, True, True, True, True])

def test_reporting():
    np.random.seed(42)
    data = 1e-7 * np.random.randn(2, 800)
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
    report = reporting(epochs, valid_labels, THR_classif, time_interval=0.25, delim=3, log={})
    classif_expected = [False, False, False, False, True, False, True, False, True, False, False,
                                           False, False, False, False, False]
    tmp = [i for k, i in enumerate(classif_expected) if valid_labels[k]]  # remove invalid
    report_expected = generate_bruxism_report(tmp, 0.25, 3)
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting2():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs

    classif_expected = [False, True, True, True, False, False]
    tmp = [i for k, i in enumerate(classif_expected) if valid_labels[k]]  # remove invalid
    report_expected = generate_bruxism_report(tmp, time_interval, delim)

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report)
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_adaptive():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    n_adaptive = 2
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs

    classif_expected = [False, True, True, False, False, False]  # loosing last because adaptive
    tmp = [i for k, i in enumerate(classif_expected) if valid_labels[k]]  # remove invalid
    report_expected = generate_bruxism_report(tmp, time_interval, delim)

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report,
                       n_adaptive=n_adaptive)
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_adaptive_forward_backward():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    n_adaptive = -2
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs

    classif_expected = [False, True, True, True, False, False]  # not loosing any because fb adaptive
    tmp = [i for k, i in enumerate(classif_expected) if valid_labels[k]]  # remove invalid
    report_expected = generate_bruxism_report(tmp, time_interval, delim)

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report,
                       n_adaptive=n_adaptive)
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)

def test_merge_labels_list():
        # Unchanging a list to the same number of elements
        v_lab = merge_labels_list([[True, False, True, False, True]], 5)
        npt.assert_equal(v_lab, [True, False, True, False, True])

        # Unchanging two identical list into one:
        v_lab = merge_labels_list([[True, False, True, False, True], [True, False, True, False, True]], 5)
        npt.assert_equal(v_lab, [True, False, True, False, True])

        # dealing with 2 coherent arrays of len(l) and 2*len(l) and getting output in 2*len(l):
        v_lab = merge_labels_list([[True, False], [True, True, False, False]], 4)
        npt.assert_equal(v_lab, [True, True, False, False])

        # dealing with classic situation:
        v_lab = merge_labels_list([[True, False], [True, True, True, True]], 4)
        npt.assert_equal(v_lab, [True, True, False, False])

        # dealing with classic situation 2:
        v_lab = merge_labels_list([[True, True], [True, True, False, True]], 4)
        npt.assert_equal(v_lab, [True, True, False, True])

        # dealing with 2 coherent arrays of len(l) and 2*len(l) and getting output in len(l):
        v_lab = merge_labels_list([[True, False], [True, True, False, False]], 2)
        npt.assert_equal(v_lab, [True, False])

        # dealing with tricky case 1:
        v_lab = merge_labels_list([[True, True], [True, True, True, False]], 2)
        npt.assert_equal(v_lab, [True, False])

        # dealing with tricky case 2:
        v_lab = merge_labels_list([[True, True, False, True], [True, True]], 2)
        npt.assert_equal(v_lab, [True, False])


def test_combine_brux_MEMA():
    labels_brux = [True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False, False, False]
    labels_artifacts_brux = [True, True, True, True, True, True, True, True,
                             True, True, True, True, True, True, True, True]
    time_interval_brux=0.25
    delim_ep_brux= 1
    labels_MEMA = [True, True, True, False,
                   False, True, True, False]
    labels_artifacts_MEMA = [True, True, True, True,
                             True, True, True, True]
    time_interval_MEMA =0.5
    delim_ep_MEMA = 1

    min_burst_joining_brux = 0
    min_burst_joining_MEMA = 0

    #classic setup
    brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA = combine_brux_MEMA\
        (labels_brux, labels_artifacts_brux, time_interval_brux, delim_ep_brux, labels_MEMA,
                      labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA,
                      min_burst_joining_brux=min_burst_joining_brux, min_burst_joining_MEMA= min_burst_joining_MEMA)


    npt.assert_equal(brux_comb_ep, [True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False, False, False])
    npt.assert_equal(MEMA_comb_ep, [True, True, True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False])

    #testing symetry of the function
    brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA = combine_brux_MEMA \
        (labels_MEMA, labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA, labels_brux, labels_artifacts_brux,
         time_interval_brux, delim_ep_brux,
         min_burst_joining_brux=min_burst_joining_MEMA, min_burst_joining_MEMA=min_burst_joining_brux)

    npt.assert_equal(MEMA_comb_ep, [True, True, True, True, False, False, False, False,
                                    True, True, True, True, False, False, False, False])
    npt.assert_equal(brux_comb_ep, [True, True, True, True, True, True, False, False, False, False,
                                    True, True, True, True, False, False])

    #Testing with inputs of same length and with one artifact affecting brux but not mema
    labels_MEMA = [True, True, True, True, False, False, False, False,
                   False, True, True, True, False, False, False, False]
    labels_artifacts_MEMA = [True, True, True, True, True, True, True, True,
                             False, True, True, True, True, True, True, True]
    time_interval_MEMA = 0.25
    delim_ep_MEMA = 1
    brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA = combine_brux_MEMA \
        (labels_MEMA, labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA, labels_brux, labels_artifacts_brux,
         time_interval_brux, delim_ep_brux,
         min_burst_joining_brux=min_burst_joining_MEMA, min_burst_joining_MEMA=min_burst_joining_brux)

    npt.assert_equal(MEMA_comb_ep, [True, True, True, True, False, False, False, False,
                                    False, False, False, False, False, False, False, False])
    npt.assert_equal(brux_comb_ep, [True, True, True, True, False, False, False, False,
                                    False, True, True, True, False, False, False, False])

    #Testing for pure events
    labels_MEMA = [True, True, True, True, False, False, False, False,
                   False, False, False, False, False, False, False, False]
    labels_artifacts_MEMA = [True, True, True, True, True, True, True, True,
                             True, True, True, True, True, True, True, True]
    labels_brux = [False, False, False, False, False, False, False, False,
                   True, True, True, True, False, False, False, False]
    labels_artifacts_brux = [True, True, True, True, True, True, True, True,
                             True, True, True, True, True, True, True, True]

    brux_comb_ep, brux_pure_ep, compt_arti_brux, MEMA_comb_ep, MEMA_pure_ep, compt_arti_MEMA = combine_brux_MEMA \
        (labels_brux, labels_artifacts_brux, time_interval_brux, delim_ep_brux, labels_MEMA,
         labels_artifacts_MEMA, time_interval_MEMA, delim_ep_MEMA,
         min_burst_joining_brux=min_burst_joining_brux, min_burst_joining_MEMA=min_burst_joining_MEMA)

    npt.assert_equal(MEMA_pure_ep, [True, True, True, True, False, False, False, False,
                                    False, False, False, False, False, False, False, False])
    npt.assert_equal(brux_pure_ep, [False, False, False, False, False, False, False, False,
                                    True, True, True, True, False, False, False, False])