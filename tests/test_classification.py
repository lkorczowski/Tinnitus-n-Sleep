import pytest
import numpy.testing as npt
from tinnsleep.classification import AmplitudeThresholding
import numpy as np


def test_AmplitudeThresholding_init():
    "test of AmplitudeThresholding initialization"
    AmplitudeThresholding()


def test_AmplitudeThresholding_init_wrong():
    "test of AmplitudeThresholding initialization"
    with pytest.raises(ValueError, match=f"`decision_function` should be callable"):
        AmplitudeThresholding(decision_function="lol")


def test_AmplitudeThresholding_values():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 0, 2
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    npt.assert_equal(classif.transform(X), X - classif.center_ * relv - absv)
    npt.assert_equal(classif.fit_transform(X), X - classif.center_ * relv - absv)

    expected_val = np.array([False, False, False, True])
    npt.assert_equal(classif.predict(X), expected_val)
    npt.assert_equal(classif.fit_predict(X), expected_val)


def test_AmplitudeThresholding_values_decision_function():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 10, 10]])
    absv, relv = 0, 2
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([1., 4., 4.]))

    npt.assert_equal(classif.transform(X), X - classif.center_ * relv - absv)
    npt.assert_equal(classif.fit_transform(X), X - classif.center_ * relv - absv)

    expected_val = np.array([False, False, False, False])
    npt.assert_equal(classif.predict(X), expected_val)
    npt.assert_equal(classif.fit_predict(X), expected_val)

    classif.decision_function = lambda distances: np.any(distances > 0, axis=-1)
    expected_val = np.array([False, False, False, True])
    npt.assert_equal(classif.fit_predict(X), expected_val)


def test_AmplitudeThresholding_values_absolute():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 4, 0
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    npt.assert_equal(classif.transform(X), X - classif.center_ * relv - absv)
    npt.assert_equal(classif.fit_transform(X), X - classif.center_ * relv - absv)

    expected_val = np.array([False, False, False, True])
    npt.assert_equal(classif.predict(X), expected_val)
    npt.assert_equal(classif.fit_predict(X), expected_val)


def test_AmplitudeThresholding_partial_fit():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 0, 2
    n_adaptive = 1
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv, n_adaptive=n_adaptive)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    for x in X:
        expected = (n_adaptive-1)/n_adaptive * classif.center_ + 1/n_adaptive * x
        classif.partial_fit(np.expand_dims(x, axis=0))
        npt.assert_equal(classif.center_, expected)


def test_AmplitudeThresholding_transform_adaptive():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 0, 2
    n_adaptive = 1
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv, n_adaptive=n_adaptive)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    # check at every step
    for x in X:
        expected = (n_adaptive-1)/n_adaptive * classif.center_ + 1/n_adaptive * x
        classif.transform(np.expand_dims(x, axis=0))
        npt.assert_equal(classif.center_, expected)

    # check macro
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv, n_adaptive=n_adaptive)
    classif.fit(X)
    classif.transform(X)
    npt.assert_equal(classif.center_, expected)


def test_AmplitudeThresholding_transform_adaptive2():
    "test of AmplitudeThresholding on test data"
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [10, 10, 10]])
    absv, relv = 0, 2
    n_adaptive = 3
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv, n_adaptive=n_adaptive)
    classif.fit(X)
    npt.assert_equal(classif.center_, np.array([4., 4., 4.]))

    # check at every step
    for x in X:
        expected = (n_adaptive-1)/n_adaptive * classif.center_ + 1/n_adaptive * x
        classif.transform(np.expand_dims(x, axis=0))
        npt.assert_equal(classif.center_, expected)

    # check macro
    classif = AmplitudeThresholding(abs_threshold=absv, rel_threshold=relv, n_adaptive=n_adaptive)
    classif.fit(X)
    classif.transform(X)
    npt.assert_equal(classif.center_, expected)
