import pytest
import numpy.testing as npt
from tinnsleep.classification import AmplitudeThresholding
import numpy as np

def test_AmplitudeThresholding_init():
    "test of AmplitudeThresholding initialization"
    AmplitudeThresholding()

def test_AmplitudeThresholding_values():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 0, 2
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    npt.assert_equal(classif.transform(X), X - classif.center_ * relv + absv)
    npt.assert_equal(classif.fit_transform(X), X - classif.center_ * relv + absv)

    expected_val = np.array([False, False, False, True])
    npt.assert_equal(classif.predict(X), expected_val)
    npt.assert_equal(classif.fit_predict(X), expected_val)
