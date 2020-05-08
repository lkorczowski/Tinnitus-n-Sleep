import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.utils import epoch
from tinnsleep.scoring import generate_bruxism_report, generate_MEMA_report
from tinnsleep.data import CreateRaw
from tinnsleep.create_reports import preprocess, reporting
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.signal import is_good_epochs, rms


def test_preprocess_unit():
    n_chan = 2
    np.random.seed(42)
    data = np.random.randn(n_chan, 400)
    duration = 50
    interval = 25
    raw = CreateRaw(data, 100, ['1', '2'])
    epochs, valid_labels, log = preprocess(raw, duration, interval)
    epochs_expected = epoch(raw.get_data(), duration, interval)
    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, [True]*epochs.shape[0])
    npt.assert_equal(log, {'suppressed_is_good': 0, 'suppressed_amp_thr': 0, 'suppressed_overall': 0,
                           'total_nb_epochs': epochs.shape[0]})

    with pytest.raises(ValueError, match=f'`filter_kwargs` a dict of parameters to pass to ``mne.raw.filter`` or None'):
        preprocess(raw, duration, interval, filter_kwargs='lol')


def test_preprocess_filter():
    n_chan = 2
    np.random.seed(42)
    data = np.random.randn(n_chan, 400)
    duration = 50
    interval = 25
    ch_names = ['1', '2']
    filter_kwargs = dict(l_freq = 20., h_freq = 40., n_jobs=4,
                         fir_design='firwin', filter_length='auto', phase='zero-double',
                         picks=ch_names)
    raw = CreateRaw(data, 100, ch_names, ch_types="misc")
    epochs, valid_labels, log = preprocess(raw, duration, interval, filter_kwargs=filter_kwargs)
    epochs_expected = epoch(raw.copy().filter(**filter_kwargs).get_data(), duration, interval)
    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, [True]*epochs.shape[0])
    npt.assert_equal(log, {'suppressed_is_good': 0, 'suppressed_amp_thr': 0, 'suppressed_overall': 0,
                           'total_nb_epochs': epochs.shape[0]})

    with pytest.raises(ValueError, match=f'`filter_kwargs` a dict of parameters to pass to ``mne.raw.filter`` or None'):
        preprocess(raw, duration, interval, filter_kwargs='lol')


def test_preprocess_is_good():
    n_chan = 4
    np.random.seed(42)
    data = np.random.randn(n_chan, 400)
    data[2, 50] = 1000    # remove one epoch
    data[3, 300:325] = 0  # remove one epoch
    data[0, slice(0, 10, 400)] = 10000  # won't be selected
    data[1, 350:400] = 0  # won't be selected
    duration = 50
    interval = 25
    raw = CreateRaw(data, 1, ['1', '2', '3', '4'])
    pick_chan = ['3', '4']
    epochs_expected = epoch(raw[pick_chan][0], duration, interval)
    #TODO: is_good params are ugly, need to refactor
    is_good_params = dict(channel_type_idx=dict(emg=[0, 1]),
                         rejection_thresholds=dict(emg=50),
                         flat_thresholds=dict(emg=1e-1))
    valid_labels_expected, _ = is_good_epochs(epochs_expected, **is_good_params)
    epochs, valid_labels, log = preprocess(raw, duration, interval,
                                            picks_chan=pick_chan, is_good_kwargs=is_good_params)
    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, valid_labels_expected)
    npt.assert_equal(log, {'suppressed_is_good': np.sum(np.invert(valid_labels_expected)),
                           'suppressed_amp_thr': 0,
                           'suppressed_overall': np.sum(np.invert(valid_labels_expected)),
                           'total_nb_epochs': epochs.shape[0]})


def test_preprocess_amp_threshold():
    n_chan = 4
    np.random.seed(42)
    data = np.random.randn(n_chan, 4000)
    data[2, slice(0, 100, 2000)] = 4    # remove one epoch
    data[3, 300:325] = 0  # remove one epoch
    data[0, slice(0, 10, 4000)] = 10000  # won't be selected
    data[1, 350:400] = 0  # won't be selected
    duration = 50
    interval = 50
    raw = CreateRaw(data, 1, ['1', '2', '3', '4'])
    pick_chan = ['3', '4']
    epochs_expected = epoch(raw[pick_chan][0], duration, interval)
    Thresholding_kwargs = dict(abs_threshold=2, rel_threshold=2, n_adaptive=0)
    epochs, valid_labels, log = preprocess(raw, duration, interval, picks_chan=pick_chan, Thresholding_kwargs=Thresholding_kwargs)

    valid_labels_expected = np.invert(AmplitudeThresholding(**Thresholding_kwargs).fit_predict(rms(epochs_expected)))
    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, valid_labels_expected)
    npt.assert_equal(log, {'suppressed_is_good': 0,
                           'suppressed_amp_thr': np.sum(np.invert(valid_labels_expected)),
                           'suppressed_overall': np.sum(np.invert(valid_labels_expected)),
                           'total_nb_epochs': epochs.shape[0]})


def test_preprocess_episode():
    with pytest.raises(ValueError, match=f"`episode_kwargs` algorithm not implemented yet"):
        n_chan = 4
        np.random.seed(42)
        data = np.random.randn(n_chan, 4000)
        data[2, slice(0, 100, 2000)] = 4  # remove one epoch
        data[3, 300:325] = 0  # remove one epoch
        data[0, slice(0, 10, 4000)] = 10000  # won't be selected
        data[1, 350:400] = 0  # won't be selected
        duration = 50
        interval = 50
        raw = CreateRaw(data, 1, ['1', '2', '3', '4'])
        pick_chan = ['3', '4']
        epochs_expected = epoch(raw[pick_chan][0], duration, interval)
        Thresholding_kwargs = dict(abs_threshold=2, rel_threshold=2, n_adaptive=0)
        episode_kwargs = dict()
        epochs, valid_labels, log = preprocess(raw, duration, interval, picks_chan=pick_chan,
                                                Thresholding_kwargs=Thresholding_kwargs,
                                                episode_kwargs=episode_kwargs)

        valid_labels_expected = np.invert(
            AmplitudeThresholding(**Thresholding_kwargs).fit_predict(rms(epochs_expected)))
        # valid_labels_expected = #TODO convert list
        npt.assert_equal(epochs, epochs_expected)
        npt.assert_equal(valid_labels, valid_labels_expected)
        npt.assert_equal(log, {'suppressed_is_good': 0,
                               'suppressed_amp_thr': np.sum(np.invert(valid_labels_expected)),
                               'suppressed_overall': np.sum(np.invert(valid_labels_expected)),
                               'total_nb_epochs': epochs.shape[0]})

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




