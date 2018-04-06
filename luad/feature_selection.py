import warnings
import numpy as np
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

base = importr('base')
base.source('lib/R/functions.R')
r_cfs_feature_idxs = robjects.globalenv['cfsFeatureIdxs']
r_fcbf_feature_idxs = robjects.globalenv['fcbfFeatureIdxs']
r_relieff_feature_idxs = robjects.globalenv['relieffFeatureIdxs']
numpy2ri.activate()

class CFS(BaseEstimator, SelectorMixin):
    """Feature selector using Correlation Feature Selection (CFS) algorithm

    Attributes
    ----------
    n_features_ : int
        Number of features in input data X

    selected_idxs_ : array-like, 1d
        CFS selected feature indexes
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        warnings.filterwarnings('ignore', category=RRuntimeWarning)
        self.selected_idxs_ = np.array(r_cfs_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self.n_features_, dtype=bool)
        mask[self.selected_idxs_] = True
        return mask

class FCBF(BaseEstimator, SelectorMixin):
    """Feature selector using Fast Correlation-Based Filter (FCBF) algorithm

    Attributes
    ----------
    n_features_ : int
        Number of features in input data X

    selected_idxs_ : array-like, 1d
        FCBF selected feature indexes
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        warnings.filterwarnings('ignore', category=RRuntimeWarning)
        self.selected_idxs_ = np.array(r_fcbf_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self.n_features_, dtype=bool)
        mask[self.selected_idxs_] = True
        return mask

class ReliefF(BaseEstimator, SelectorMixin):
    """Feature selector using ReliefF algorithm

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Feature scores
    """
    def __init__(self, k=20, thres=0, n_neighbors=15, sample_size=10):
        self.k = k
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        warnings.filterwarnings('ignore', category=RRuntimeWarning)
        self.scores_ = np.array(r_relieff_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = np.zeros(self.scores_.shape, dtype=bool)
        mask[np.argsort(self.scores_, kind="mergesort")[-self.k:]] = True
        return mask
