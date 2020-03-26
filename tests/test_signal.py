import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms, create_basic_detection
from tinnsleep.data import CreateRaw, RawToEpochs_sliding

def test_rms():
    X = np.array([
        [[1, 2, 3],
         [1, 2, 3]],
        [[1, 3, 3],
         [2, 4, 10]]
    ])
    rms_values = rms(X)
    npt.assert_almost_equal(rms_values, np.array([[1., 2., 3.], [1.58113883, 3.53553391, 7.38241153]]), decimal=4)
    
def test_create_basic_detection():
    data = np.random.randn(2, 4000)
    ch_names = ['Fz', 'Pz']
    duration = 0.1 #in seconds
    interval =0.1  #in seconds
    raw= CreateRaw(data, ch_names)
    #print(raw.get_data())
    epo = RawToEpochs_sliding(raw, duration, interval, picks=None).get_data()
    stack_rms=(rms(epo, ax=2))
    print(create_basic_detection(stack_rms, 0, 1))
   
    