import pytest
import numpy.testing as npt
from tinnsleep.scoring import classif_to_burst, burst_to_episode, create_list_events, rearrange_chronological
from tinnsleep.burst import burst


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


def test_create_list_events():
    # Test if empty
    npt.assert_equal(create_list_events([], 0.5), [])

    # Test all the episodes types
    bursty = [burst(0.1, 1), burst(0.1, 1), burst(3, 6), burst(1.5, 2.5), burst(1.25, 3.5), burst(0, 0.2),
              burst(5.5, 6.5), burst(7.5, 8.5), burst(15, 20), burst(25, 26), burst(26.5, 27), burst(28, 29)]

    li_ep = burst_to_episode(bursty)
    li_ev = create_list_events(li_ep, 0.25)
    npt.assert_equal(li_ev, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                             3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    # Test with empty initial inputs
    bursty = [burst(1, 4)]
    li_ep = burst_to_episode(bursty)
    li_ev = create_list_events(li_ep, 0.25)
    npt.assert_equal(li_ev, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
