import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from ..univariate_selection import BaseScorer

def _score_features(srv, X, y):
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        scores[j] = srv.fit(X[:, [j]], y).score(X[:, [j]], y)
    return scores


class CoxnetScorerSurvival(BaseScorer):
    """Coxnet feature scorer for survival tasks.

    Parameters
    ----------
    l1_ratio : float, optional (default=0.5)
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1. For l1_ratio = 0
        the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For
        0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    normalize : boolean, optional (default=False)
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm. If you wish to
        standardize, please use sklearn.preprocessing.StandardScaler before
        calling fit on an estimator with normalize=False.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature c-index scores.
    """
    def __init__(self, l1_ratio=0.5, normalize=False):
        self.l1_ratio = l1_ratio
        self.normalize = normalize

    def fit(self, X, y):
        """Run scorer on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self._check_params(X, y)
        self.scores_ = _score_features(
            CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, normalize=self.normalize), X, y
        )
        return self

    def _check_params(self, X, y):
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError(
                "l1_ratio should be between 0 and 1; got %r."
                % self.l1_ratio
            )


class CoxPHScorerSurvival(BaseScorer):
    """CoxPH feature scorer for survival tasks.

    Parameters
    ----------
    alpha : float, optional (default=0)
        Regularization parameter for ridge regression penalty.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature c-index scores.
    """
    def __init__(self, alpha=0):
        self.alpha = alpha

    def fit(self, X, y):
        """Run scorer on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self._check_params(X, y)
        self.scores_ = _score_features(
            CoxPHSurvivalAnalysis(alpha=self.alpha), X, y
        )
        return self

    def _check_params(self, X, y):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "alpha should be between 0 and 1; got %r."
                % self.alpha
            )


class IPCRidgeScorerSurvival(BaseScorer):
    """IPCRidge feature scorer for survival tasks.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Small positive values of alpha improve the conditioning of the
        problem and reduce the variance of the estimates.

    normalize : boolean, optional (default=False)
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm. If you wish to
        standardize, please use sklearn.preprocessing.StandardScaler before
        calling fit on an estimator with normalize=False.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature c-index scores.
    """
    def __init__(self, alpha=1.0, normalize=False):
        self.alpha = alpha
        self.normalize = normalize

    def fit(self, X, y):
        """Run scorer on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self._check_params(X, y)
        self.scores_ = _score_features(
            IPCRidge(alpha=self.alpha, normalize=self.normalize), X, y
        )
        return self

    def _check_params(self, X, y):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "alpha should be between 0 and 1; got %r."
                % self.alpha
            )
