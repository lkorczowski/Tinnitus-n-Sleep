"""Module for classification function."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import softmax

class AmplitudeThresholding(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Thresholding

    The classifier estimate the centroid value of a given signal (class 0) and detect
    observation that reach a the value above `abs_threshold + abs_threshold * centroid`

    Parameters
    ----------
    abs_threshold : float, (default: 0.)
        absolute value to add to the centroid for thresholding
    abs_threshold : float, (default: 2.)
        multiplicator factor of the centroid for thresholding

    Attributes
    ----------
    center_ : ndarray, shape (n_features, )

    See Also
    --------

    """

    def __init__(self, abs_threshold=0., rel_threshold=2.):
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    def fit(self, X, y=[], sample_weight=None):
        """Compute average smplitude

                Parameters
                ----------
                X : {array-like} of shape (n_samples, n_features)

                y : Ignored
                    Not used, present here for API consistency by convention.

                sample_weight : array-like of shape (n_samples,), default=None
                    The weights for each observation in X. If None, all observations
                    are assigned equal weight.
                Returns
                -------
                self
                    Fitted estimator.
                """

        # Check that X and y have correct shape
        X = check_array(X)

        self.center_ = np.average(X, axis=0, weights=sample_weight)

        # Return the classifier
        # La petite branche est passée là
        return self


    def _predict_distances(self, X):
        distances = np.nan * np.ones(X.shape)
        for k in range(X.shape[0]):
            distances[k, :] = X[k, :] - (self.center_ * self.rel_threshold) + self.abs_threshold
        return distances

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        distances = self._predict_distances(X)

        return np.all(distances > 0, axis=1)

    def transform(self, X):
        """get the distance to each centroid.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y=[]):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)
