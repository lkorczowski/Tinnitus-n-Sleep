import pytest
import numpy.testing as npt
from tinnsleep.scoring import classif_to_burst, burst_to_episode
from tinnsleep.burst import burst



def test_classif_to_burst():
    
    li=[True,False,True, False, False, True, True, False, True, True, True, 
        True, False, False, False, False, False, True,True, True, False, True]
    interval=1
    li_burst=classif_to_burst(li,interval)
    npt.assert_equal(len(li_burst), 6)
    
def test_burst_to_episode():
    
    bursty=[burst(0.1,1)]
    bursty.append(burst(0.1,1))
    bursty.append(burst(3,6))
    bursty.append(burst(1.5,2.5))
    bursty.append(burst(1.25,3.5))
    bursty.append(burst(0,0.2))
    bursty.append(burst(5.5,6.5)) 
    bursty.append(burst(15,20)) 
    
    li_ep=burst_to_episode(bursty)
    
    npt.assert_equal(len(li_ep), 2)
    npt.assert_equal(li_ep[0].beg, 0)
    npt.assert_equal(li_ep[0].end, 6.5)
    npt.assert_equal(len(li_ep[0].burst_list), 2)
    npt.assert_equal(li_ep[0].is_tonic, True)
    
    
    
test_classif_to_burst()
test_burst_to_episode()