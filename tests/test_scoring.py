import pytest
import numpy.testing as npt
from tinnsleep.scoring import classif_to_burst, burst_to_episode, create_list_events, rearrange_chronological, \
    generate_clinical_report,  generate_annotations, generate_MEMA_report
from tinnsleep.burst import burst
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

    npt.assert_equal(len(burst_to_episode(bursty,  delim=1, min_burst_joining=3)), 1)
    npt.assert_equal(len(burst_to_episode(bursty, delim=3, min_burst_joining=3)), 2)
    npt.assert_equal(len(burst_to_episode(bursty, delim=1, min_burst_joining=4)), 1)




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
    npt.assert_equal(create_list_events([], 0.5, 0), [])

    # Test all the episodes types
    bursty = [burst(0.1, 1), burst(0.1, 1), burst(3, 6), burst(1.5, 2.5), burst(1.25, 3.5), burst(0, 0.2),
              burst(5.5, 6.5), burst(7.5, 8.5), burst(15, 20), burst(25, 26), burst(26.5, 27), burst(28, 29)]

    li_ep = burst_to_episode(bursty)
    anno = generate_annotations(li_ep)
    npt.assert_equal(len(anno), 3)
    li_ev = create_list_events(li_ep, 0.25, 29)
    npt.assert_equal(li_ev, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                             3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


    # Test with empty initial inputs and ending
    bursty = [burst(1, 4)]
    li_ep = burst_to_episode(bursty)
    li_ev = create_list_events(li_ep, 0.25, 5)
    npt.assert_equal(li_ev, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

def test_generate_clinical_report():
    classif = [False, False]
    report = generate_clinical_report(classif, 1)
    npt.assert_equal( report["Number of bursts per episode"], 0)

    classif = [True, False, True, False, False, True, True, False, True, True, True,
          True, False, False, False, False, False, True, True, True, False, True]
    report = generate_clinical_report(classif, 1)
    npt.assert_equal(len(report), 13)
    npt.assert_equal(report["Mean duration of mixed episode"], 12.0)
    npt.assert_equal(report["Mean duration of phasic episode"], np.nan)
    npt.assert_equal(report["Number of bursts per episode"], 3.0)


    classif.extend([False, False, False, False, False, False, True, False, False, True, False, False, True])
    report = generate_clinical_report(classif, 1)
    npt.assert_equal(len(report), 13)
    npt.assert_equal(report["Mean duration of phasic episode"], 7.0)
    npt.assert_equal(report["Total burst duration"], 15)


def test_generate_MEMA_report():
    classif = [False, False]
    report = generate_MEMA_report(classif, 1)
    npt.assert_equal( report["Number of MEMA bursts per episode"], 0)

    classif = [True, True, True, False, False, True, True, True]
    report = generate_MEMA_report(classif, 1)
    npt.assert_equal(len(report), 8)
    npt.assert_equal(report["Total number of MEMA burst"], 2)
    npt.assert_equal(report["Mean duration of MEMA episode"], 8.0)
    npt.assert_equal(report["Number of MEMA bursts per episode"], 2.0)


    classif.extend([False, False, False, False, False, False, True, False, False, True, False, False, True])
    report = generate_MEMA_report(classif, 1)
    npt.assert_equal(report["Mean duration of MEMA episode"], 7.5)
    npt.assert_equal(report["Total MEMA burst duration"], 9)


