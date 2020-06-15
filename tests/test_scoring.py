import numpy.testing as npt
from tinnsleep.events.scoring import classif_to_burst, burst_to_episode, create_list_events, rearrange_chronological,\
    generate_annotations, episodes_to_list
from tinnsleep.events.burst import burst
import numpy as np


def test_classif_to_burst():
    li = [True, False, True, False, False, True, True, False, True, True, True,
          True, False, False, False, False, False, True, True, True, False, True]
    interval = 1
    li_burst = classif_to_burst(li, interval)
    npt.assert_equal(len(li_burst), 6)


def test_empty_classif_to_burst():
    li = []
    interval = 1
    li_burst = classif_to_burst(li, interval)
    npt.assert_equal(len(li_burst), 0)


def test_empty_burst_to_episode():
    li = []
    li_ep = burst_to_episode(li)
    npt.assert_equal(len(li_ep), 0)


def test_burst_to_episode():
    bursty = [burst(0.1, 1), burst(0.1, 1), burst(3, 6), burst(1.5, 2.5), burst(1.25, 3.5), burst(0, 0.2),
              burst(5.5, 6.5), burst(15, 20)]
    li_ep = burst_to_episode(bursty)

    npt.assert_equal(len(li_ep), 2)
    npt.assert_equal(li_ep[0].beg, 0)
    npt.assert_equal(li_ep[0].end, 6.5)
    npt.assert_equal(len(li_ep[0].burst_list), 2)
    npt.assert_equal(li_ep[0].is_tonic, True)


def test_burst_to_episode_withparam():
    bursty = [burst(3.1, 3.9), burst(5, 6),
              burst(6.5, 7.5), burst(15, 20)]

    npt.assert_equal(len(burst_to_episode(bursty,  delim=0, min_burst_joining=1)), 4)
    npt.assert_equal(len(burst_to_episode(bursty,  delim=1, min_burst_joining=3)), 1)
    npt.assert_equal(len(burst_to_episode(bursty, delim=3, min_burst_joining=3)), 2)
    npt.assert_equal(len(burst_to_episode(bursty, delim=1, min_burst_joining=4)), 1)

    bursty = [burst(1, 2), burst(3, 4)]
    npt.assert_equal(len(burst_to_episode(bursty,  delim=0, min_burst_joining=1)), 2)
    

def test_rearrange_chronological():
    """Test if a given burst_list is in the chronological order"""
    bursty = [burst(0.1, 1), burst(0.1, 1), burst(3, 6), burst(1.5, 2.5)]
    bursty = rearrange_chronological(bursty)
    flag = True
    for i in range(len(bursty) - 1):
        if bursty[i + 1].beg < bursty[i].beg:
            flag = False
            break
    npt.assert_equal(flag, True)


def test_episodes_to_list_simple():
    time_interval = 1
    bursts_ = []
    n_labels = 10
    labels = episodes_to_list(burst_to_episode(bursts_, min_burst_joining=1, delim=0), time_interval, n_labels)
    npt.assert_equal(labels, [False]*10)

    bursts_ = [burst(1, 2), burst(5, 6), burst(6, 7), burst(9, 10)]
    labels = episodes_to_list(burst_to_episode(bursts_, min_burst_joining=1, delim=0), time_interval, n_labels)
    labels_expected = np.zeros((n_labels,));
    labels_expected[[1, 5, 6, 9]] = True
    npt.assert_equal(labels>0, labels_expected)

    bursts_ = [burst(0.5, 2), burst(4.5, 7.5),burst(8.1, 8.2), burst(9, 10.5)] # should convert only full epochs
    labels = episodes_to_list(burst_to_episode(bursts_, min_burst_joining=1, delim=0), time_interval, n_labels)
    labels_expected = np.zeros((n_labels,));
    labels_expected[[1, 5, 6, 9]] = True
    npt.assert_equal(labels>0, labels_expected)
