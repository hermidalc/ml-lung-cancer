"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from sklearn.feature_selection.mutual_info_ import (
    mutual_info_regression, mutual_info_classif
)
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from .univariate_selection import (
    chi2, f_classif, f_oneway, f_regression, SelectPercentile,
    SelectKBest, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect,
    ANOVAFScorerClassification, ANOVAFScorerRegression, Chi2Scorer,
    MutualInfoScorerClassification, MutualInfoScorerRegression
)
from .from_model import SelectFromModel
from .rfe import RFE, RFECV
from .custom import (
    CFS, ColumnSelector, FCBF, LimmaScorerClassification, ReliefF
)


__all__ = ['chi2',
           'f_classif',
           'f_oneway',
           'f_regression',
           'mutual_info_classif',
           'mutual_info_regression',
           'SelectFdr',
           'SelectFpr',
           'SelectFwe',
           'SelectKBest',
           'SelectFromModel',
           'SelectPercentile',
           'GenericUnivariateSelect',
           'RFE',
           'RFECV',
           'VarianceThreshold',
           'ANOVAFScorerClassification',
           'ANOVAFScorerRegression',
           'Chi2Scorer',
           'MutualInfoScorerClassification',
           'MutualInfoScorerRegression',
           'CFS',
           'ColumnSelector',
           'FCBF',
           'LimmaScorerClassification',
           'ReliefF']
