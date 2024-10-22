import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.utils import epoch, label_report
from tinnsleep.events.scoring import burst_to_episode, classif_to_burst, episodes_to_list
from tinnsleep.data import CreateRaw
from tinnsleep.reports import preprocess, reporting, combine_brux_MEMA, generate_bruxism_report,\
    generate_MEMA_report
from tinnsleep.classification import AmplitudeThresholding
from tinnsleep.signal import is_good_epochs, rms, power_ratio


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
                           'total_nb_epochs': epochs.shape[0], 'suppressed_ratio': 0})

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
                           'total_nb_epochs': epochs.shape[0], 'suppressed_ratio': 0})

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
                           'total_nb_epochs': epochs.shape[0],
                           'suppressed_ratio': np.sum(np.invert(valid_labels_expected))/epochs.shape[0]})


def test_preprocess_amp_threshold():
    n_chan = 4
    np.random.seed(42)
    data = np.random.randn(n_chan, 4000)
    data[2, 150:200] = 5    # remove one epoch
    data[3, 250:350] = 5    # remove two epochs
    data[0, slice(0, 10, 4000)] = 10000  # won't be selected
    data[1, 350:400] = 0                 # won't be selected
    duration = 50
    interval = duration
    raw = CreateRaw(data, 1, ['1', '2', '3', '4'])
    pick_chan = ['3', '4']
    epochs_expected = epoch(raw[pick_chan][0], duration, interval)
    Thresholding_kwargs = dict(abs_threshold=2, rel_threshold=2, n_adaptive=0,
                               decision_function=lambda foo: np.any(foo>0, axis=-1))
    epochs, valid_labels, log = preprocess(raw, duration, interval, picks_chan=pick_chan, Thresholding_kwargs=Thresholding_kwargs)

    valid_labels_expected = np.invert(AmplitudeThresholding(**Thresholding_kwargs).fit_predict(rms(epochs_expected)))
    assert np.any(valid_labels_expected) # pytest preliminary verification

    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, valid_labels_expected)
    npt.assert_equal(log, {'suppressed_is_good': 0,
                           'suppressed_amp_thr': 3,
                           'suppressed_overall': 3,
                           'total_nb_epochs': epochs.shape[0],
                           'suppressed_ratio': np.sum(np.invert(valid_labels_expected))/epochs.shape[0]})


def test_preprocess_episode():
    n_chan = 4
    np.random.seed(42)
    data = np.random.randn(n_chan, 4000)
    data[2, 150:200] = 5    # remove one epoch
    data[3, 250:350] = 5    # remove two epochs
    data[0, slice(0, 10, 4000)] = 10000  # won't be selected
    data[1, 350:400] = 0                 # won't be selected
    duration = 50
    interval = duration
    raw = CreateRaw(data, 1, ['1', '2', '3', '4'])
    pick_chan = ['3', '4']
    epochs_expected = epoch(raw[pick_chan][0], duration, interval)
    Thresholding_kwargs = dict(abs_threshold=2, rel_threshold=2, n_adaptive=0,
                               decision_function=lambda foo: np.any(foo>0, axis=-1))
    valid_labels_expected = AmplitudeThresholding(**Thresholding_kwargs).fit_predict(rms(epochs_expected))
    assert np.any(valid_labels_expected) # pytest preliminary verification

    burst_to_episode_kwargs = dict()
    epochs, valid_labels, log = preprocess(raw, duration, interval, picks_chan=pick_chan,
                                            Thresholding_kwargs=Thresholding_kwargs,
                                            burst_to_episode_kwargs=burst_to_episode_kwargs)

    time_interval = interval / raw.info["sfreq"]
    valid_labels_expected = 0 < episodes_to_list(
                burst_to_episode(
                    classif_to_burst(valid_labels_expected, time_interval=time_interval),
                    **burst_to_episode_kwargs
                ), time_interval=time_interval, n_labels=valid_labels_expected.shape[0]
            )
    valid_labels_expected = np.invert(valid_labels_expected)
    npt.assert_equal(epochs, epochs_expected)
    npt.assert_equal(valid_labels, valid_labels_expected)
    npt.assert_equal(log, {'suppressed_is_good': 0,
                           'suppressed_amp_thr': np.sum(np.invert(valid_labels_expected)),
                           'suppressed_overall': np.sum(np.invert(valid_labels_expected)),
                           'total_nb_epochs': epochs.shape[0],
                           'suppressed_ratio': np.sum(np.invert(valid_labels_expected))/epochs.shape[0]})


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
    classif_expected = np.array([False, False, False, False, True, False, True, False, True, False, False,
                                           False, False, False, False, False])
    report_expected = generate_bruxism_report(classif_expected[valid_labels], 0.25, 3, valid_labels)
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
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

    classif_expected = np.array([False, True, True, True, False, False])
    report_expected = generate_bruxism_report(classif_expected[valid_labels], time_interval, delim,  valid_labels)

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report)
    npt.assert_equal(report["labels"][0], classif_expected)
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_bruxism_with_sleep():
    np.random.seed(42)
    data = 1e-7 * np.random.randn(2, 800)
    data[0][100:150] += 100 # epoch 2 (also bad epoch)
    data[1][100:150] += 100
    data[0][200:250] += 100 # epoch 4
    data[1][200:250] += 100
    data[0][300:350] += 100 # epoch 6
    data[1][300:350] += 100
    data[0][400:450] += 100 # epoch 8
    data[1][400:450] += 100

    duration = 50
    interval = 50

    epochs = epoch(data, duration, interval, axis=-1)
    THR_classif = [[0, 2], [0, 3]]
    valid_labels = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    sleep_labels = ["REM"] * len(epochs)
    sleep_labels[10] = "awake"
    valid_labels[10] = False
    sleep_labels[1] = "N2"
    sleep_labels[2] = "N3"
    sleep_labels = np.array(sleep_labels)

    report = reporting(epochs, valid_labels, THR_classif, time_interval=0.25, delim=3, log={}, sleep_labels=sleep_labels)
    classif_expected = np.array([False, False, False, False, True, False, True, False, True, False, False,
                                           False, False, False, False, False])

    report_expected = generate_bruxism_report(classif_expected[valid_labels], 0.25, 3, valid_labels, sleep_labels=sleep_labels[valid_labels])
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    report_expected.update(label_report(sleep_labels))
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_bruxism_with_sleep_params():
    np.random.seed(42)
    data = 1e-7 * np.random.randn(2, 800)
    data[0][100:150] += 100 # epoch 2 (also bad epoch)
    data[1][100:150] += 100
    data[0][200:250] += 100 # epoch 4
    data[1][200:250] += 100
    data[0][300:350] += 100 # epoch 6
    data[1][300:350] += 100
    data[0][400:450] += 100 # epoch 8
    data[1][400:450] += 100

    duration = 50
    interval = 50

    epochs = epoch(data, duration, interval, axis=-1)
    THR_classif = [[0, 2], [0, 3]]
    valid_labels = [True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True]
    sleep_labels = ["REM"] * len(epochs)
    sleep_labels[10] = "awake"
    valid_labels[10] = False
    sleep_labels[1] = "N2"
    sleep_labels[2] = "N3"
    sleep_labels[4] = "awake"

    sleep_labels = np.array(sleep_labels)

    report = reporting(epochs, valid_labels, THR_classif, time_interval=0.25, delim=3, log={}, sleep_labels=sleep_labels)
    classif_expected = np.array([False, False, False, False, False, False, True, False, True, False, False,
                                           False, False, False, False, False])

    report_expected = generate_bruxism_report(classif_expected[valid_labels], 0.25, 3, valid_labels, sleep_labels=sleep_labels[valid_labels])
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    report_expected.update(label_report(sleep_labels))
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_MEMA_with_sleep():
    np.random.seed(42)
    data = 1e-7 * np.random.randn(2, 800)
    data[0][100:150] += 100 # epoch 2 (also bad epoch)
    data[1][100:150] += 100
    data[0][200:250] += 100 # epoch 4
    data[1][200:250] += 100
    data[0][300:350] += 100 # epoch 6
    data[1][300:350] += 100
    data[0][400:450] += 100 # epoch 8
    data[1][400:450] += 100

    duration = 50
    interval = 50

    epochs = epoch(data, duration, interval, axis=-1)
    THR_classif = [[0, 2], [0, 3]]
    valid_labels = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
    sleep_labels = ["REM"] * len(epochs)
    sleep_labels[10] = "awake"
    valid_labels[10] = False
    sleep_labels[1] = "N2"
    sleep_labels[2] = "N3"
    sleep_labels = np.array(sleep_labels)

    report = reporting(epochs, valid_labels, THR_classif, time_interval=0.25, delim=3, log={},
                       generate_report=generate_MEMA_report, sleep_labels=sleep_labels)
    classif_expected = np.array([False, False, False, False, True, False, True, False, True, False, False,
                                           False, False, False, False, False])

    report_expected = generate_MEMA_report(classif_expected[valid_labels], 0.25, 3, valid_labels, sleep_labels=sleep_labels[valid_labels])
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    report_expected.update(label_report(sleep_labels))
    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["reports"][0], report_expected)


def test_reporting_sleep_all_invalid():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs
    sleep_labels = np.array([False, True])

    THR_classif = [[0, 1.5], [0, 1.6]]
    with pytest.raises(ValueError):
        report = reporting(epochs, valid_labels, THR_classif,
                           time_interval=time_interval, delim=delim, log={},
                           generate_report=generate_bruxism_report,
                            sleep_labels=sleep_labels)


def test_reporting_adaptive_bruxism():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    n_adaptive = 2
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs

    classif_expected = np.array([False, True, True, False, False, False])  # loosing last because adaptive
    classif_expected2 = np.array([False, True, False, False, False, False])  # loosing last because adaptive

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report,
                       n_adaptive=n_adaptive)
    report_expected = generate_bruxism_report(classif_expected[valid_labels], time_interval, delim, valid_labels)
    report_expected2 = generate_bruxism_report(classif_expected2[valid_labels], time_interval, delim, valid_labels)

    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["labels"][1], classif_expected2)
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    report_expected2["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected2[valid_labels])

    npt.assert_equal(report["reports"][0], report_expected)
    npt.assert_equal(report["reports"][1], report_expected2)


def test_reporting_adaptive_MEMA():
    np.random.seed(42)
    n_epochs = 6
    duration = 3
    time_interval = 0.25
    n_adaptive = 2
    delim = 3
    epochs = np.random.randn(n_epochs, 2, duration)
    epochs[1:4:] += 100
    valid_labels = [True] * n_epochs

    classif_expected = np.array([False, True, True, False, False, False])  # loosing last because adaptive
    classif_expected2 = np.array([False, True, False, False, False, False])  # loosing last because adaptive

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_MEMA_report,
                       n_adaptive=n_adaptive)
    report_expected = generate_MEMA_report(classif_expected[valid_labels], time_interval, delim, valid_labels)
    report_expected2 = generate_MEMA_report(classif_expected2[valid_labels], time_interval, delim, valid_labels)

    npt.assert_equal(report["labels"][0], classif_expected)
    npt.assert_equal(report["labels"][1], classif_expected2)
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    report_expected2["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected2[valid_labels])

    npt.assert_equal(report["reports"][0], report_expected)
    npt.assert_equal(report["reports"][1], report_expected2)


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

    classif_expected = np.array([False, True, True, True, False, False])  # not loosing any because fb adaptive
    report_expected = generate_bruxism_report(classif_expected[valid_labels], time_interval, delim, valid_labels)

    THR_classif = [[0, 1.5], [0, 1.6]]
    report = reporting(epochs, valid_labels, THR_classif,
                       time_interval=time_interval, delim=delim, log={},
                       generate_report=generate_bruxism_report,
                       n_adaptive=n_adaptive)
    npt.assert_equal(report["labels"][0], classif_expected)
    report_expected["Power Ratio"] = power_ratio(epochs[valid_labels], classif_expected[valid_labels])
    npt.assert_equal(report["reports"][0], report_expected)


def test_combine_brux_MEMA():
    labels_brux = [True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False, False, False]

    time_interval_brux=0.25
    delim_ep_brux= 1
    labels_MEMA = [True, True, True, False,
                   False, True, True, False]

    time_interval_MEMA =0.5
    delim_ep_MEMA = 1

    min_burst_joining_brux = 0
    min_burst_joining_MEMA = 0

    #classic setup
    brux_comb_ep, brux_pure_ep,  MEMA_comb_ep, MEMA_pure_ep = combine_brux_MEMA\
        (labels_brux,  time_interval_brux, delim_ep_brux, labels_MEMA,
                       time_interval_MEMA, delim_ep_MEMA,
                      min_burst_joining_brux=min_burst_joining_brux, min_burst_joining_MEMA= min_burst_joining_MEMA)


    npt.assert_equal(brux_comb_ep, [True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False, False, False])
    npt.assert_equal(MEMA_comb_ep, [True, True, True, True, True, True, False, False, False, False,
                   True, True, True, True, False, False])

    #testing symetry of the function
    brux_comb_ep, brux_pure_ep, MEMA_comb_ep, MEMA_pure_ep = combine_brux_MEMA \
        (labels_MEMA, time_interval_MEMA, delim_ep_MEMA, labels_brux,
         time_interval_brux, delim_ep_brux,
         min_burst_joining_brux=min_burst_joining_MEMA, min_burst_joining_MEMA=min_burst_joining_brux)

    npt.assert_equal(MEMA_comb_ep, [True, True, True, True, False, False, False, False,
                                    True, True, True, True, False, False, False, False])
    npt.assert_equal(brux_comb_ep, [True, True, True, True, True, True, False, False, False, False,
                                    True, True, True, True, False, False])


    #Testing for pure events
    labels_MEMA = [True, True, True, True, False, False, False, False,
                   False, False, False, False, False, False, False, False]

    labels_brux = [False, False, False, False, False, False, False, False,
                   True, True, True, True, False, False, False, False]


    brux_comb_ep, brux_pure_ep, MEMA_comb_ep, MEMA_pure_ep = combine_brux_MEMA \
        (labels_brux,  time_interval_brux, delim_ep_brux, labels_MEMA,
         time_interval_MEMA, delim_ep_MEMA,
         min_burst_joining_brux=min_burst_joining_brux, min_burst_joining_MEMA=min_burst_joining_MEMA)

    npt.assert_equal(MEMA_pure_ep, [True, True, True, True, False, False, False, False,
                                    False, False, False, False, False, False, False, False])
    npt.assert_equal(brux_pure_ep, [False, False, False, False, False, False, False, False,
                                    True, True, True, True, False, False, False, False])


def test_generate_bruxism_report():
    classif = [False, False]
    valid_labels = [True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal( report["Number of bursts per episode"], 0)

    classif = [True, False, True, False, False, True, True, False, True, True, True,
          True, False, False, False, False, False, True, True, True, False, True]
    valid_labels = [True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(len(report), 13)

    npt.assert_equal(report["Clean data duration"], len(classif))
    npt.assert_equal(report["Total burst duration"], 12)
    npt.assert_equal(report["Total number of burst"], 6)
    npt.assert_almost_equal(report["Number of bursts per hour"], 3600/len(classif)*6)
    npt.assert_equal(report["Total number of episodes"], 2)
    npt.assert_equal(report["Number of bursts per episode"], 3.0)
    npt.assert_almost_equal(report["Number of episodes per hour"], 3600/len(classif)*2)
    npt.assert_almost_equal(report["Number of tonic episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of mixed episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of phasic episodes per hour"], 3600/len(classif)*0)
    npt.assert_almost_equal(report["Mean duration of mixed episode"], 12.0)
    npt.assert_almost_equal(report["Mean duration of phasic episode"], np.nan)
    npt.assert_almost_equal(report["Mean duration of tonic episode"], 5)


    classif.extend([False, False, False, False, False, False, True, False, False, True, False, False, True])
    #valid_labels.extend([False, False, False, False, False, False, True, False, False, True, False, False, True])
    valid_labels.extend([True, True, True, True, True, True, True, True, True, True, True, True, True])
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(len(report), 13)
    npt.assert_equal(report["Clean data duration"], len(classif))
    npt.assert_equal(report["Total burst duration"], 15)
    npt.assert_equal(report["Total number of burst"], 9)
    npt.assert_almost_equal(report["Number of bursts per hour"], 3600/len(classif)*9)
    npt.assert_equal(report["Total number of episodes"], 3)
    npt.assert_equal(report["Number of bursts per episode"], 3.0)
    npt.assert_almost_equal(report["Number of episodes per hour"], 3600/len(classif)*3)
    npt.assert_almost_equal(report["Number of tonic episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of mixed episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of phasic episodes per hour"], 3600/len(classif)*1)
    npt.assert_equal(report["Mean duration of mixed episode"], 12)
    npt.assert_equal(report["Mean duration of phasic episode"], 7.0)
    npt.assert_equal(report["Mean duration of tonic episode"], 5)

    #Case with all labels valid
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, True, True, True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of episodes"], 1)
    #Case with one invalid labels, the episode should be suppressed
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, False, True, True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of episodes"], 0)
    #Case with an invalid label just before the episode, the episode should be conserved
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, False, True, True, True, True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of episodes"], 1)
    #Case with an invalid labels on the edge of the end of the episode, the episode should be suppressed
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, True, True, True, False]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of episodes"], 0)



def test_generate_MEMA_report():
    classif = [False, False]
    valid_labels = [True, True]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal( report["Number of MEMA bursts per episode"], 0)

    classif = [True, True, True, False, False, True, True, True]
    valid_labels = [True, True, True, True, True, True, True, True, True]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(len(report), 8)
    npt.assert_equal(report["Clean MEMA duration"], len(classif))
    npt.assert_equal(report["Total MEMA burst duration"], 6)
    npt.assert_equal(report["Number of MEMA bursts per hour"], 3600/len(classif)*2)
    npt.assert_equal(report["Total number of MEMA episodes"], 1)
    npt.assert_equal(report["Number of MEMA bursts per episode"], 2.0)
    npt.assert_equal(report["Number of MEMA episodes per hour"], 3600/len(classif)*1)
    npt.assert_equal(report["Total number of MEMA burst"], 2)
    npt.assert_equal(report["Mean duration of MEMA episode"], len(classif))


    classif.extend([False, False, False, False, False, False, True, False, False, True, False, False, True])
    valid_labels.extend([True, True, True, True, True, True, True, True, True, True, True, True, True])
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(len(report), 8)
    npt.assert_equal(report["Clean MEMA duration"], len(classif))
    npt.assert_equal(report["Total MEMA burst duration"], 9)
    npt.assert_equal(report["Number of MEMA bursts per hour"], 3600/len(classif)*5)
    npt.assert_equal(report["Total number of MEMA episodes"], 2)
    npt.assert_equal(report["Number of MEMA bursts per episode"], 2.5)
    npt.assert_equal(report["Number of MEMA episodes per hour"], 3600/len(classif)*2)
    npt.assert_equal(report["Total number of MEMA burst"], 5)
    npt.assert_equal(report["Mean duration of MEMA episode"], 7.5)

    # Case with all labels valid
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, True, True, True, True]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of MEMA episodes"], 1)
    # Case with one invalid labels, the episode should be suppressed
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, False, True, True, True]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of MEMA episodes"], 0)
    # Case with an invalid label just before the episode, the episode should be conserved
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, False, True, True, True, True, True]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of MEMA episodes"], 1)
    # Case with an invalid labels on the edge of the end of the episode, the episode should be suppressed
    classif = [False, False, True, False, True, False, True]
    valid_labels = [True, True, True, True, True, True, False]
    report = generate_MEMA_report(classif, 1, 3, valid_labels)
    npt.assert_equal(report["Total number of MEMA episodes"], 0)


def test_generate_bruxism_report2():
    classif = [False, False, False, True, True, True]
    valid_labels = [True, True, True, True, True, True]
    report = generate_bruxism_report(classif, 1, 3, valid_labels)
    npt.assert_equal(len(report), 13)

    npt.assert_equal(report["Clean data duration"], len(classif))
    npt.assert_equal(report["Total burst duration"], 3)
    npt.assert_equal(report["Total number of burst"], 1)
    npt.assert_almost_equal(report["Number of bursts per hour"], 3600/len(classif)*1)
    npt.assert_equal(report["Total number of episodes"], 1)
    npt.assert_equal(report["Number of bursts per episode"], 1)
    npt.assert_almost_equal(report["Number of episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of tonic episodes per hour"], 3600/len(classif)*1)
    npt.assert_almost_equal(report["Number of mixed episodes per hour"], 3600/len(classif)*0)
    npt.assert_almost_equal(report["Number of phasic episodes per hour"], 3600/len(classif)*0)
    npt.assert_almost_equal(report["Mean duration of mixed episode"], np.nan)
    npt.assert_almost_equal(report["Mean duration of phasic episode"], np.nan)
    npt.assert_almost_equal(report["Mean duration of tonic episode"], 3)


def test_generate_MEMA_report_with_sleep():
    classif = [False, False, False, True, True, True]
    valid_labels = [True, True, True, True, True, True]
    sleep_labels = ["awake", "REM", "awake", "awake", "REM", "REM"]
    report = generate_MEMA_report(classif, 1, 3,valid_labels, sleep_labels=sleep_labels)
    npt.assert_equal(report['MEMA episode awake count'], 1)
    npt.assert_equal(report['MEMA episode awake ratio'], 1.0)


def test_generate_bruxism_report_with_sleep():
    classif = [False, False, False, True, True, True]
    valid_labels = [True, True, True, True, True, True]
    sleep_labels = ["awake", "REM", "awake", "awake", "REM", "REM"]
    report = generate_bruxism_report(classif, 1, 3,valid_labels, sleep_labels=sleep_labels)
    npt.assert_equal(report['bruxism episode awake count'], 1)
    npt.assert_equal(report['bruxism episode awake ratio'], 1.0)
    npt.assert_equal(report['bruxism episode awake ratio'], 1.0)


def test_generate_MEMA_report_with_sleep_mismatch():
    classif = [False, False, False, True, True, True]
    valid_labels = [True, True, True, True, True, True]
    sleep_labels = ["awake", "REM", "awake", "awake", "REM"]
    with pytest.raises(ValueError,
                       match=f"classif is \({len(classif)},\), sleep_labels is \({len(sleep_labels)},\)"):
        generate_MEMA_report(classif, 1, 3, valid_labels, sleep_labels=sleep_labels)


def test_generate_bruxism_report_with_sleep_mismatch():
    classif = [False, False, False, True, True, True]
    valid_labels = [True, True, True,True, True, True]
    sleep_labels = ["awake", "REM", "awake", "awake", "REM"]
    with pytest.raises(ValueError,
                       match=f"classif is \({len(classif)},\), sleep_labels is \({len(sleep_labels)},\)"):
        generate_bruxism_report(classif, 1, 3, valid_labels, sleep_labels=sleep_labels)
