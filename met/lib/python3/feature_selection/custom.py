import warnings
import numpy as np
import rpy2.robjects as robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.externals.joblib import Memory
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from .univariate_selection import BaseScorer

base = importr('base')
base.source('functions.R')
r_limma_feature_score = robjects.globalenv['limma_feature_score']
r_cfs_feature_idxs = robjects.globalenv['cfs_feature_idxs']
r_fcbf_feature_idxs = robjects.globalenv['fcbf_feature_idxs']
r_relieff_feature_score = robjects.globalenv['relieff_feature_score']
numpy2ri.activate()

def fcbf_feature_idxs(X, y, threshold=0):
    idxs, scores = r_fcbf_feature_idxs(X, y, threshold=threshold)
    return np.array(idxs, dtype=int), np.array(scores, dtype=float)

def relieff_feature_score(X, y):
    return np.array(r_relieff_feature_score(X, y), dtype=float)


class LimmaScorerClassification(BaseScorer):
    """Limma feature scorer for classification tasks.

    Parameters
    ----------
    pkm : bool, default=False
        If X matrix is RNA-seq FPKM/RPKM/TPM normalized count data

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        The set of F values.

    pvalues_ : array, shape (n_features,)
        The set of p-values.
    """
    def __init__(self, pkm=False):
        self.pkm = pkm

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
        f, pv = r_limma_feature_score(X, y, pkm=self.pkm)
        self.scores_, self.pvalues_ = np.array(f, dtype=float), np.array(pv, dtype=float)
        return self

    def _check_params(self, X, y):
        pass


class ColumnSelector(BaseEstimator, SelectorMixin):
    """Manual column feature selector

    Parameters
    ----------
    cols : array-like (default=None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns.
        If None, returns all columns in the array.

    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).
    """
    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

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
        ---------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self._check_params(X, y)
        self._n_features = X.shape[1]
        return self

    def _check_params(self, X, y):
        if self.cols is not None:
            for col in self.cols:
                if not 0 <= col <= X.shape[1]:
                    raise ValueError(
                        "cols should be 0 <= col <= n_features; got %r."
                        "Use cols=None to return all features."
                        % col
                    )

    def _get_support_mask(self):
        check_is_fitted(self, '_n_features')
        if self.cols is None:
            mask = np.ones(self._n_features, dtype=bool)
        else:
            mask = np.zeros(self._n_features, dtype=bool)
            mask[list(self.cols)] = True
        return mask


class CFS(BaseEstimator, SelectorMixin):
    """Feature selector using Correlation Feature Selection (CFS) algorithm

    Attributes
    ----------
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
        self._n_features = X.shape[1]
        warnings.filterwarnings('ignore', category=RRuntimeWarning, message="^Rjava\.init\.warning")
        self.selected_idxs_ = np.array(r_cfs_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        mask[self.selected_idxs_] = True
        return mask


class FCBF(BaseEstimator, SelectorMixin):
    """Feature selector using Fast Correlation-Based Filter (FCBF) algorithm

    Parameters
    ----------
    k : int or "all", optional, default="all"
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.
        If k is specified threshold is ignored.

    threshold : float, optional default=0
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded.

    memory : None, str or object with the joblib.Memory interface, optional \
        (default=None)
    Used for internal caching. By default, no caching is done.
    If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    selected_idxs_ : array-like, 1d
        FCBF selected feature indexes
    """
    def __init__(self, k='all', threshold=0, memory=None):
        self.k = k
        self.threshold = threshold
        self.memory = memory
        self.selected_idxs_ = np.array([], dtype=int)
        self.scores_ = np.array([], dtype=float)
        self._n_features = None

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
        self._check_params(X, y)
        memory = self.memory
        if memory is None:
            memory = Memory(cachedir=None, verbose=0)
        elif isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)
        elif not isinstance(memory, Memory):
            raise ValueError(
                "'memory' should either be a string or"
                " a sklearn.externals.joblib.Memory"
                " instance, got 'memory={!r}' instead."
                .format(type(memory))
            )
        self._n_features = X.shape[1]
        if self.k == 'all' or self.k > 0:
            warnings.filterwarnings('ignore', category=RRuntimeWarning, message="^Rjava\.init\.warning")
            feature_idxs, scores = memory.cache(fcbf_feature_idxs)(X, y, threshold=self.threshold)
            warnings.filterwarnings('always', category=RRuntimeWarning)
            if self.k != 'all':
                feature_idxs = feature_idxs[np.argsort(scores, kind='mergesort')[-self.k:]]
                scores = np.sort(scores, kind='mergesort')[-self.k:]
            self.selected_idxs_ = np.sort(feature_idxs, kind='mergesort')
            self.scores_ = scores[np.argsort(feature_idxs, kind='mergesort')]
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k
            )

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        if self.k == 'all' or self.k > 0:
            mask[self.selected_idxs_] = True
        return mask


class ReliefF(BaseEstimator, SelectorMixin):
    """Feature selector using ReliefF algorithm

    Parameters
    ----------
    k : int or "all", optional, default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.
        If k is specified threshold is ignored.

    threshold : float, optional default=0
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded.

    n_neighbors : int, optional default=20
        Number of neighbors for ReliefF algorithm

    sample_size : int, optional default=10
        Sample size for ReliefF algorithm

    memory : None, str or object with the joblib.Memory interface, optional \
        (default=None)
    Used for internal caching. By default, no caching is done.
    If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Feature scores
    """
    def __init__(self, k=10, threshold=0, n_neighbors=20, sample_size=10, memory=None):
        self.k = k
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.memory = memory

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
        self._check_params(X, y)
        memory = self.memory
        if memory is None:
            memory = Memory(cachedir=None, verbose=0)
        elif isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)
        elif not isinstance(memory, Memory):
            raise ValueError(
                "'memory' should either be a string or"
                " a sklearn.externals.joblib.Memory"
                " instance, got 'memory={!r}' instead."
                .format(type(memory))
            )
        warnings.filterwarnings('ignore', category=RRuntimeWarning, message="^Rjava\.init\.warning")
        self.scores_ = memory.cache(relieff_feature_score)(X, y)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k
            )

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.scores_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.scores_, kind='mergesort')[-self.k:]] = True
        return mask
