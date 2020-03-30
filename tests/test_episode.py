import pytest
import numpy.testing as npt
from tinnsleep.burst import burst
from tinnsleep.episode import episode


def test_episode():
    bursty=burst(0.1,1)
    bursty2=burst(0.1,1)
    bursty3=burst(3,6)
    bursty4=burst(1.5,2.5)
    bursty5=burst(1.25,3.5) 
    bursty6=burst(0,0.2) 
    bursty7=burst(5.5,6.5) 
    
    
    #We try with one basic burst
    epi=episode(bursty)
    npt.assert_equal(epi.beg, 0.1)
    npt.assert_equal(epi.end, 1)
    npt.assert_equal(epi.nb_burst, 1)
    npt.assert_equal(len(epi.burst_list), 1)
    epi.set_tonic()
    epi.set_phasic()
    epi.set_mixed()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    npt.assert_equal(epi.is_valid(), False)
    
    
    #We try to add the same burst
    epi.add_a_burst(bursty2)
    npt.assert_equal(epi.beg, 0.1)
    npt.assert_equal(epi.end, 1)
    npt.assert_equal(epi.nb_burst, 1)
    npt.assert_equal(len(epi.burst_list), 1)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    npt.assert_equal(epi.is_valid(), False)
    
    #We try to add a new burst (tonic) afterwards
    epi.add_a_burst(bursty3)
    npt.assert_equal(epi.beg,0.1)
    npt.assert_equal(epi.end, 6)
    npt.assert_equal(epi.nb_burst, 2)
    npt.assert_equal(len(epi.burst_list), 2)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, True)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    npt.assert_equal(epi.is_valid(), True)
    
    #We try to add a burst in the middle
    epi.add_a_burst(bursty4)
    npt.assert_equal(epi.beg, 0.1)
    npt.assert_equal(epi.end, 6)
    npt.assert_equal(epi.nb_burst, 3)
    npt.assert_equal(len(epi.burst_list), 3)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, False)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, True)
    npt.assert_equal(epi.is_valid(), True)
    
    #We try to add a new burst, overlapping with burst 1 and 2
    npt.assert_equal(epi.burst_list[1].beg, 1.5)
    npt.assert_equal(epi.burst_list[1].end, 2.5)
    epi.add_a_burst(bursty5)
    npt.assert_equal(epi.beg, 0.1)
    npt.assert_equal(epi.end, 6)
    npt.assert_equal(epi.nb_burst, 2)
    npt.assert_equal(len(epi.burst_list), 2)
    npt.assert_equal(epi.burst_list[1].beg, 1.25)
    npt.assert_equal(epi.burst_list[1].end, 6)
    epi.assess_type()
    npt.assert_equal(epi.is_tonic, True)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    
    # We try to add a burst before the first one and overlapping with it
    epi.add_a_burst(bursty6)
    npt.assert_equal(epi.beg, 0)
    npt.assert_equal(epi.end, 6)
    npt.assert_equal(epi.nb_burst, 2)
    npt.assert_equal(len(epi.burst_list), 2)
    epi.set_tonic()
    epi.set_phasic()
    epi.set_mixed()
    npt.assert_equal(epi.is_tonic, True)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    
    #We try to add a burst after the last one and overlapping with it
    epi.add_a_burst(bursty7)
    npt.assert_equal(epi.beg, 0)
    npt.assert_equal(epi.end, 6.5)
    npt.assert_equal(epi.nb_burst, 2)
    npt.assert_equal(len(epi.burst_list), 2)
    epi.set_tonic()
    epi.set_phasic()
    epi.set_mixed()
    npt.assert_equal(epi.is_tonic, True)
    npt.assert_equal(epi.is_phasic, False)
    npt.assert_equal(epi.is_mixed, False)
    npt.assert_equal(epi.is_valid(), True)
    
    #We verify the final burst parameters
    npt.assert_equal(epi.burst_list[0].beg, 0)
    npt.assert_equal(epi.burst_list[0].end, 1)
    npt.assert_equal(epi.burst_list[1].beg, 1.25)
    npt.assert_equal(epi.burst_list[1].end, 6.5)
    
test_episode()