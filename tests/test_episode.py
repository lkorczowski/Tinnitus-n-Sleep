import pytest
import numpy.testing as npt
from tinnsleep.burst import burst
from tinnsleep.episode import episode


def test_episode():
    bursty = burst(0.1, 1)
    epi = episode(bursty)
    npt.assert_equal(epi.beg, 0.1)
    npt.assert_equal(epi.end, 1)
    npt.assert_equal(len(epi.burst_list), 1)

def test_overlap_left():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(1.75, 3.5)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 1)

def test_overlapx2_left():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(4.75, 5.5)
    bursty3 = burst(3, 3.9)
    bursty4 = burst(2, 3.4)
    epi = episode(bursty)
    epi.add_a_burst(bursty2)
    epi.add_a_burst(bursty3)
    epi.add_a_burst(bursty4)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 5.5)
    npt.assert_equal(len(epi.burst_list), 2)


def test_no_overlap_left():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(2.75, 3.5)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 2)


def test_overlap_right():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(1.75, 3.5)
    epi = episode(bursty)
    epi.add_a_burst(bursty2)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 1)

def test_overlap_right_2_bursts():
    bursty = burst(1.5, 2.5)
    bursty3 = burst(2.6, 2.9)
    bursty2 = burst(2.75, 3.5)
    epi = episode(bursty)
    epi.add_a_burst(bursty3)
    epi.add_a_burst(bursty2)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 2)


def test_overlapx2_right():
    bursty = burst(1.5, 2.5)
    bursty3 = burst(2, 2.9)
    bursty2 = burst(2.75, 3.5)
    epi = episode(bursty)
    epi.add_a_burst(bursty2)
    epi.add_a_burst(bursty3)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 1)


def test_no_overlap_right():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(2.75, 3.5)
    epi = episode(bursty)
    epi.add_a_burst(bursty2)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 3.5)
    npt.assert_equal(len(epi.burst_list), 2)


def test_megaburst():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(1.75, 3.5)
    bursty3 = burst(1.4, 3.6)
    epi = episode(bursty)
    epi.add_a_burst(bursty2)
    epi.add_a_burst(bursty3)
    npt.assert_equal(epi.beg, 1.4)
    npt.assert_equal(epi.end, 3.6)
    npt.assert_equal(len(epi.burst_list), 1)


def test_burst_already_there():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(1.5, 2.5)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    npt.assert_equal(epi.beg, 1.5)
    npt.assert_equal(epi.end, 2.5)
    npt.assert_equal(len(epi.burst_list), 1)

def test_overlap_inside():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(4.75, 5.5)
    bursty3 = burst(3, 3.9)
    bursty4 = burst(2.7, 3.9)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    epi.add_a_burst(bursty3)
    epi.add_a_burst(bursty4)
    npt.assert_equal(epi.burst_list[1].beg, 2.7)
    npt.assert_equal(epi.burst_list[1].end, 3.9)
    npt.assert_equal(len(epi.burst_list), 3)

def test_overlapx2_inside():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(4.75, 5.5)
    bursty3 = burst(3, 3.9)
    bursty4 = burst(2.7, 2.9)
    bursty5 = burst(2.8, 3.7)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    epi.add_a_burst(bursty3)
    epi.add_a_burst(bursty4)
    epi.add_a_burst(bursty5)
    npt.assert_equal(epi.burst_list[1].beg, 2.7)
    npt.assert_equal(epi.burst_list[1].end, 3.9)
    npt.assert_equal(len(epi.burst_list), 3)

def test_no_overlap_inside():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(4.75, 5.5)
    bursty3 = burst(3, 3.9)
    bursty4 = burst(2.7, 2.9)
    epi = episode(bursty2)
    epi.add_a_burst(bursty)
    epi.add_a_burst(bursty3)
    epi.add_a_burst(bursty4)
    npt.assert_equal(len(epi.burst_list), 4)


def test_assess_type_is_valid():
    bursty = burst(1.5, 2.5)
    bursty2 = burst(4, 6.5)
    bursty3 = burst(3, 3.5)
    bursty4 = burst(2.7, 2.9)

    #Test tonic
    epi1 = episode(bursty2)
    epi1.assess_type()
    npt.assert_equal(epi1.is_tonic, True)
    npt.assert_equal(epi1.is_phasic, False)
    npt.assert_equal(epi1.is_mixed, False)
    npt.assert_equal(epi1.is_valid(), True)

    # Reassessment invariant
    epi1.assess_type()
    npt.assert_equal(epi1.is_tonic, True)
    npt.assert_equal(epi1.is_phasic, False)
    npt.assert_equal(epi1.is_mixed, False)
    npt.assert_equal(epi1.is_valid(), True)

    #Test mixed
    epi1.add_a_burst(bursty3)
    epi1.add_a_burst(bursty4)
    epi1.assess_type()
    npt.assert_equal(epi1.is_tonic, False)
    npt.assert_equal(epi1.is_phasic, False)
    npt.assert_equal(epi1.is_mixed, True)

    #Test no type 1 burst
    epi = episode(bursty)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    npt.assert_equal(epi.is_valid(), False)

    #Test no type 2 burst
    epi.add_a_burst(bursty3)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)

    #Test phasic
    epi.add_a_burst(bursty4)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, True)
    npt.assert_equal(epi.is_mixed, False)
