"""Module for classification function."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_array


class AmplitudeThresholding(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Thresholding

    The classifier estimate the centroid value of a given signal (class 0) and detect
    observation that reach a the value above `abs_threshold + rel_threshold * centroid`

    Parameters
    ----------
    abs_threshold : float, (default: 0.)
        absolute value to add to the centroid for thresholding
    rel_threshold : float, (default: 2.)
        multiplicator factor of the centroid for thresholding
    adaptive_n : int, (default: 0)
        number of past observations used to update the centroid during transform by moving average.
        If 0, then no update is applied.
        .. math:: \bar{X}_k = \frac{n-1}{n}(\bar(X)_{k-1} + \frac{1}{n}X_{k-1}
    decision_function: function (default: numpy.all(distances>0))
        function used in predict to convert multidimensional distance features into label.
        by default, `numpy.all(distances>0)` is used: True if all features distance are greater than 0, i.e.
        >>> decision_function = lambda distances: np.all(distances>0, axis=-1)  # default
        if `numpy.any` True if any features for a given trial is greater than the threshold
        >>> decision_function = lambda distances: np.any(distances>0, axis=-1)  # valid decision function
        any function take a ndarray of boolean and the `axis` parameters can be used
        for example, the following function threshold the distances while keeping same dimension (no merge)
        >>> decision_function = lambda distances: distances>0  # build a function identity that doesn't merge the labels

    Attributes
    ----------
    center_ : ndarray, shape (n_features, )

    See Also
    --------

    """

    def __init__(self, abs_threshold=0., rel_threshold=2., n_adaptive=0, decision_function=None):
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold
        self.n_adaptive = n_adaptive
        if not callable(decision_function):
            if decision_function is None:
                decision_function = lambda foofoo: np.all(foofoo > 0, axis=-1)
            else:
                raise ValueError("`decision_function` should be callable")
        self.decision_function = decision_function


    def fit(self, X, y=[], sample_weight=None):
        """Compute average amplitude

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
        return self

    def partial_fit(self, X, y=[]):
        """Updating average amplitude for adaptive thresholding

                Parameters
                ----------
                X : {array-like} of shape (n_samples, n_features)

                y : Ignored
                    Not used, present here for API consistency by convention.

                Returns
                -------
                self
                    Fitted estimator.
                """

        # Check that X and y have correct shape
        X = check_array(X)
        sample_weight = [(self.n_adaptive-1)/self.n_adaptive, 1/self.n_adaptive]
        for x in X:
            self.center_ = np.average(np.array([self.center_, x]), axis=0, weights=sample_weight)

        # Return the classifier
        return self

    def _predict_distances(self, X):
        distances = np.nan * np.ones(X.shape)
        for k in range(X.shape[0]):
            distances[k, :] = X[k, :] - ((self.center_ * self.rel_threshold) + self.abs_threshold)
        return distances

    def predict(self, X):

        # Input validation
        X = check_array(X)
        distances = self.transform(X)

        return self.decision_function(distances)

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
        if self.n_adaptive == 0:
            d = self._predict_distances(X)
        else:
            d = np.zeros(X.shape)
            for (k, x) in enumerate(X):
                x = np.expand_dims(x, axis=0)
                d[k, :] = self._predict_distances(x)
                self.partial_fit(x)
        return d

    def fit_predict(self, X, y=[]):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)
