import pytest
import numpy.testing as npt
from tinnsleep.burst import burst


def test_burst():
    
    bursty=burst(0,1)
    bursty2=burst(0,1)
    bursty3=burst(1,3)
    bursty4=burst(0.25,1.75)
    
    npt.assert_equal(bursty.is_equal(bursty2),True) 
    npt.assert_equal(bursty.is_equal(bursty3),False)
    npt.assert_equal(bursty.is_before(bursty2),False)
    npt.assert_equal(bursty.is_before(bursty3),True)
    npt.assert_equal(bursty.is_tonic,False)
    npt.assert_equal(bursty3.is_tonic,True)
    
    npt.assert_equal(bursty.is_overlapping(bursty2), True)
    npt.assert_equal(bursty4.is_overlapping(bursty3), True)
    npt.assert_equal(bursty4.is_overlapping(bursty), True)
    npt.assert_equal(bursty.is_overlapping(bursty3), False)
    
    npt.assert_equal(bursty.merge_if_overlap(bursty2), True)
    npt.assert_equal(bursty4.merge_if_overlap(bursty3), True)
    npt.assert_equal(bursty4.merge_if_overlap(bursty), True)
    npt.assert_equal(bursty.merge_if_overlap(bursty3), False)
    