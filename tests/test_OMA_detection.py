import pytest
import numpy as np
from tinnsleep.data import CreateRaw, RawToEpochs_sliding
from tinnsleep.OMA_detection import OMA_thresholding_sliding
import numpy.testing as npt


def test_OMA_thresholding_sliding():
    np.random.seed(42)
    data = np.random.randn(1, 400)
    for i in range(100):
        data[0][i+200] += 100
    duration = 50
    interval = 50
    THR = [0,2]
    OMA_labels = OMA_thresholding_sliding(data, duration, interval, THR)
    npt.assert_equal(OMA_labels, [True, True, True, True, False, False, True, True])

