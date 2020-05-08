import pytest
import numpy.testing as npt
from tinnsleep.scoring import classif_to_burst, burst_to_episode, create_list_events, rearrange_chronological, \
    generate_bruxism_report,  generate_annotations, generate_MEMA_report
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

    npt.assert_equal(len(burst_to_episode(bursty,  delim=0, min_burst_joining=1)), 4)
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

def test_generate_bruxism_report():
    classif = [False, False]
    report = generate_bruxism_report(classif, delim=3, time_interval=1)
    npt.assert_equal( report["Number of bursts per episode"], 0)

    classif = [True, False, True, False, False, True, True, False, True, True, True,
          True, False, False, False, False, False, True, True, True, False, True]
    report = generate_bruxism_report(classif, 1, 3)
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
    report = generate_bruxism_report(classif, 1, 3)
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


def test_generate_MEMA_report():
    classif = [False, False]
    report = generate_MEMA_report(classif, 1, delim=3)
    npt.assert_equal( report["Number of MEMA bursts per episode"], 0)

    classif = [True, True, True, False, False, True, True, True]
    report = generate_MEMA_report(classif, 1, delim=3)
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
    report = generate_MEMA_report(classif, 1, delim=3)
    npt.assert_equal(len(report), 8)
    npt.assert_equal(report["Clean MEMA duration"], len(classif))
    npt.assert_equal(report["Total MEMA burst duration"], 9)
    npt.assert_equal(report["Number of MEMA bursts per hour"], 3600/len(classif)*5)
    npt.assert_equal(report["Total number of MEMA episodes"], 2)
    npt.assert_equal(report["Number of MEMA bursts per episode"], 2.5)
    npt.assert_equal(report["Number of MEMA episodes per hour"], 3600/len(classif)*2)
    npt.assert_equal(report["Total number of MEMA burst"], 5)
    npt.assert_equal(report["Mean duration of MEMA episode"], 7.5)


def test_generate_bruxism_report2():
    classif = [False, False, False, True, True, True]
    report = generate_bruxism_report(classif, 1, 3)
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

def test_generate_report_fails_noparams():
    """check if the parameters are required or test fails"""
    classif = [False, False]
    with pytest.raises(TypeError, match=f"missing 2 required positional arguments: 'time_interval' and 'delim'"):
        generate_bruxism_report(classif)
    with pytest.raises(TypeError, match=f"missing 2 required positional arguments: 'time_interval' and 'delim'"):
        generate_MEMA_report(classif)
