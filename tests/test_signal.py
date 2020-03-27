import pytest
import numpy as np
import numpy.testing as npt
from tinnsleep.signal import rms


def test_rms():
    X = np.array([
        [[1, 2, 3],
         [1, 2, 3]],
        [[1, 3, 3],
         [2, 4, 10]],
        [[0, 0, 1],
          [1, 0, 0]]
    ])
    rms_values = rms(X)
    npt.assert_almost_equal(rms_values, np.array([[2.1602469,  2.1602469 ],[2.51661148, 6.32455532],[0.57735027, 0.57735027]]), decimal=4)

