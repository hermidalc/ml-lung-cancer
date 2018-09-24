#!/usr/bin/env python

import warnings
from argparse import ArgumentParser
from copy import deepcopy
from os import makedirs, path
from pprint import pprint
from tempfile import mkdtemp
from shutil import rmtree
from natsort import natsorted
from itertools import combinations
from operator import itemgetter
import numpy as np
import rpy2.rinterface as rinterface
rinterface.set_initoptions((b'rpy2', b'--quiet', b'--no-save', b'--max-ppsize=500000'))
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
# from rpy2.robjects import pandas2ri
# import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import (
    chi2, f_classif, mutual_info_classif, SelectKBest, SelectFpr, SelectFromModel, RFE, VarianceThreshold
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import delayed, dump, Memory, Parallel
from feature_selection import CFS, FCBF, ReliefF
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

# ignore QDA collinearity warnings
warnings.filterwarnings('ignore', category=UserWarning, message="^Variables are collinear")

# config
parser = ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--test-splits', type=int, default=10, help='num outer splits')
parser.add_argument('--test-size', type=float, default=0.3, help='outer splits test size')
parser.add_argument('--dataset-tr', type=str, nargs='+', help='dataset tr')
parser.add_argument('--dataset-te', type=str, nargs='+', help='dataset te')
parser.add_argument('--num-combo-tr', type=int, default=1, help='dataset tr num combos')
parser.add_argument('--data-type', type=str, nargs='+', help='dataset data type')
parser.add_argument('--norm-meth', type=str, nargs='+', help='normalization method')
parser.add_argument('--feat-type', type=str, nargs='+', help='dataset feature type')
parser.add_argument('--prep-meth', type=str, nargs='+', help='dataset preprocess method')
parser.add_argument('--bc-meth', type=str, nargs='+', help='batch effect correction method')
parser.add_argument('--filt-type', type=str, nargs='+', help='dataset filter type')
parser.add_argument('--filt-zero-feats', default=False, action='store_true', help='filter zero features')
parser.add_argument('--filt-zero-samples', default=False, action='store_true', help='filter zero samples')
parser.add_argument('--filt-zsdv-feats', default=False, action='store_true', help='filter zero sd features')
parser.add_argument('--filt-zsdv-samples', default=False, action='store_true', help='filter zero sd samples')
parser.add_argument('--filt-nzvr-feats', default=False, action='store_true', help='filter near zero var features')
parser.add_argument('--filt-nzvr-feat-freq-cut', type=float, help='filter near zero var features freq cutoff')
parser.add_argument('--filt-nzvr-feat-uniq-cut', type=int, help='filter near zero var features uniq cutoff')
parser.add_argument('--filt-nzvr-samples', default=False, action='store_true', help='filter near zero var samples')
parser.add_argument('--filt-nzvr-sample-freq-cut', type=float, help='filter near zero var sample freq cutoff')
parser.add_argument('--filt-nzvr-sample-uniq-cut', type=int, help='filter near zero var sample uniq cutoff')
parser.add_argument('--filt-ncor-feats', default=False, action='store_true', help='filter non corr features')
parser.add_argument('--filt-ncor-feat-cut', type=float, help='filter non corr feature cutoff')
parser.add_argument('--filt-ncor-samples', default=False, action='store_true', help='filter non corr samples')
parser.add_argument('--filt-ncor-sample-cut', type=float, help='filter non corr sample cutoff')
parser.add_argument('--no-addon-te', default=False, action='store_true', help='no add-on dataset te')
parser.add_argument('--fs-meth', type=str, nargs='+', help='feature selection method')
parser.add_argument('--slr-meth', type=str, nargs='+', help='scaling method')
parser.add_argument('--clf-meth', type=str, nargs='+', help='classifier method')
parser.add_argument('--slr-mms-fr-min', type=int, nargs='+', help='slr mms fr min')
parser.add_argument('--slr-mms-fr-max', type=int, nargs='+', help='slr mms fr max')
parser.add_argument('--fs-vrt-thres', type=float, nargs='+', help='fs vrt threshold')
parser.add_argument('--fs-skb-k', type=int, nargs='+', help='fs skb k select')
parser.add_argument('--fs-skb-k-max', type=int, default=1000, help='fs skb k select max')
parser.add_argument('--fs-skb-lim-off', default=False, action='store_true', help='skb turn off sample limit')
parser.add_argument('--fs-sfp-p', type=float, nargs='+', help='fs sfp fpr')
parser.add_argument('--fs-sfm-svm-thres', type=float, nargs='+', help='fs sfm svm threshold')
parser.add_argument('--fs-sfm-svm-c', type=float, nargs='+', help='fs sfm svm c')
parser.add_argument('--fs-sfm-svm-cw', type=str, nargs='+', help='fs sfm svm class weight')
parser.add_argument('--fs-sfm-ext-thres', type=float, nargs='+', help='fs sfm ext threshold')
parser.add_argument('--fs-sfm-ext-e', type=int, nargs='+', help='fs sfm ext n estimators')
parser.add_argument('--fs-sfm-ext-e-max', type=int, default=50, help='fs sfm ext n estimators max')
parser.add_argument('--fs-sfm-ext-d', type=int, nargs='+', help='fs sfm ext max depth')
parser.add_argument('--fs-sfm-ext-d-max', type=int, default=10, help='fs sfm ext max depth max')
parser.add_argument('--fs-sfm-ext-f', type=str, nargs='+', help='fs sfm ext max features')
parser.add_argument('--fs-sfm-ext-cw', type=str, nargs='+', help='fs sfm ext class weight')
parser.add_argument('--fs-sfm-grb-e', type=int, nargs='+', help='fs sfm grb n estimators')
parser.add_argument('--fs-sfm-grb-e-max', type=int, default=200, help='fs sfm grb n estimators max')
parser.add_argument('--fs-sfm-grb-d', type=int, nargs='+', help='fs sfm grb max depth')
parser.add_argument('--fs-sfm-grb-d-max', type=int, default=10, help='fs sfm grb max depth max')
parser.add_argument('--fs-sfm-grb-f', type=str, nargs='+', help='fs sfm grb max features')
parser.add_argument('--fs-rfe-svm-c', type=float, nargs='+', help='fs rfe svm c')
parser.add_argument('--fs-rfe-svm-cw', type=str, nargs='+', help='fs rfe svm class weight')
parser.add_argument('--fs-rfe-ext-e', type=int, nargs='+', help='fs rfe ext n estimators')
parser.add_argument('--fs-rfe-ext-e-max', type=int, default=50, help='fs rfe ext n estimators max')
parser.add_argument('--fs-rfe-ext-d', type=int, nargs='+', help='fs rfe ext max depth')
parser.add_argument('--fs-rfe-ext-d-max', type=int, default=10, help='fs rfe ext max depth max')
parser.add_argument('--fs-rfe-ext-f', type=str, nargs='+', help='fs rfe ext max features')
parser.add_argument('--fs-rfe-ext-cw', type=str, nargs='+', help='fs rfe ext class weight')
parser.add_argument('--fs-rfe-grb-e', type=int, nargs='+', help='fs rfe grb n estimators')
parser.add_argument('--fs-rfe-grb-e-max', type=int, default=200, help='fs rfe grb n estimators max')
parser.add_argument('--fs-rfe-grb-d', type=int, nargs='+', help='fs rfe grb max depth')
parser.add_argument('--fs-rfe-grb-d-max', type=int, default=10, help='fs rfe grb max depth max')
parser.add_argument('--fs-rfe-grb-f', type=str, nargs='+', help='fs rfe grb max features')
parser.add_argument('--fs-rfe-step', type=float, nargs='+', help='fs rfe step')
parser.add_argument('--fs-rfe-verbose', type=int, default=0, help='fs rfe verbosity')
parser.add_argument('--fs-rlf-n', type=int, nargs='+', help='fs rlf n neighbors')
parser.add_argument('--fs-rlf-n-max', type=int, default=20, help='fs rlf n neighbors max')
parser.add_argument('--fs-rlf-s', type=int, nargs='+', help='fs rlf sample size')
parser.add_argument('--fs-rlf-s-max', type=int, default=10, help='fs rlf sample size max')
parser.add_argument('--fs-rank-meth', type=str, default='mean_weights', help='fs rank method')
parser.add_argument('--clf-svm-c', type=float, nargs='+', help='clf svm c')
parser.add_argument('--clf-svm-cw', type=str, nargs='+', help='clf svm class weight')
parser.add_argument('--clf-svm-kern', type=str, nargs='+', help='clf svm kernel')
parser.add_argument('--clf-svm-deg', type=int, nargs='+', help='clf svm poly degree')
parser.add_argument('--clf-svm-g', type=float, nargs='+', help='clf svm gamma')
parser.add_argument('--clf-svm-cache', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--clf-knn-k', type=int, nargs='+', help='clf knn neighbors')
parser.add_argument('--clf-knn-k-max', type=int, default=10, help='clf knn neighbors max')
parser.add_argument('--clf-knn-w', type=str, nargs='+', help='clf knn weights')
parser.add_argument('--clf-dt-d', type=int, nargs='+', help='clf dt max depth')
parser.add_argument('--clf-dt-d-max', type=int, default=10, help='clf dt max depth max')
parser.add_argument('--clf-dt-f', type=str, nargs='+', help='clf dt max features')
parser.add_argument('--clf-dt-cw', type=str, nargs='+', help='clf dt class weight')
parser.add_argument('--clf-rf-e', type=int, nargs='+', help='clf rf n estimators')
parser.add_argument('--clf-rf-e-max', type=int, default=50, help='clf rf n estimators max')
parser.add_argument('--clf-rf-d', type=int, nargs='+', help='clf rf max depth')
parser.add_argument('--clf-rf-d-max', type=int, default=10, help='clf rf max depth max')
parser.add_argument('--clf-rf-f', type=str, nargs='+', help='clf rf max features')
parser.add_argument('--clf-rf-cw', type=str, nargs='+', help='clf rf class weight')
parser.add_argument('--clf-ext-e', type=int, nargs='+', help='clf ext n estimators')
parser.add_argument('--clf-ext-e-max', type=int, default=50, help='clf ext n estimators max')
parser.add_argument('--clf-ext-d', type=int, nargs='+', help='clf ext max depth')
parser.add_argument('--clf-ext-d-max', type=int, default=10, help='clf ext max depth max')
parser.add_argument('--clf-ext-f', type=str, nargs='+', help='clf ext max features')
parser.add_argument('--clf-ext-cw', type=str, nargs='+', help='clf ext class weight')
parser.add_argument('--clf-ada-e', type=int, nargs='+', help='clf ada n estimators')
parser.add_argument('--clf-ada-e-max', type=int, default=200, help='clf ada n estimators max')
parser.add_argument('--clf-ada-lgr-c', type=float, nargs='+', help='clf ada lgr c')
parser.add_argument('--clf-ada-lgr-cw', type=str, nargs='+', help='clf ada lgr class weight')
parser.add_argument('--clf-grb-e', type=int, nargs='+', help='clf grb n estimators')
parser.add_argument('--clf-grb-e-max', type=int, default=200, help='clf grb n estimators max')
parser.add_argument('--clf-grb-d', type=int, nargs='+', help='clf grb max depth')
parser.add_argument('--clf-grb-d-max', type=int, default=10, help='clf grb max depth max')
parser.add_argument('--clf-grb-f', type=str, nargs='+', help='clf grb max features')
parser.add_argument('--clf-mlp-hls', type=str, nargs='+', help='clf mlp hidden layer sizes')
parser.add_argument('--clf-mlp-act', type=str, nargs='+', help='clf mlp activation function')
parser.add_argument('--clf-mlp-slvr', type=str, nargs='+', help='clf mlp solver')
parser.add_argument('--clf-mlp-a', type=float, nargs='+', help='clf mlp alpha')
parser.add_argument('--clf-mlp-lr', type=str, nargs='+', help='clf mlp learning rate')
parser.add_argument('--scv-type', type=str, default='grid', help='scv type (grid or rand)')
parser.add_argument('--scv-splits', type=int, default=100, help='scv splits')
parser.add_argument('--scv-size', type=float, default=0.3, help='scv size')
parser.add_argument('--scv-verbose', type=int, default=1, help='scv verbosity')
parser.add_argument('--scv-refit', type=str, default='roc_auc', help='scv refit score func (roc_auc, bcr)')
parser.add_argument('--scv-n-iter', type=int, default=100, help='randomized scv num iterations')
parser.add_argument('--scv-h-plt-meth', type=str, default='best', help='scv hyperparam plot meth (best, all)')
parser.add_argument('--show-annots', type=str, nargs='+', help='show annotations')
parser.add_argument('--save-figs', default=False, action='store_true', help='save figures')
parser.add_argument('--show-figs', default=False, action='store_true', help='show figures')
parser.add_argument('--save-model', default=False, action='store_true', help='save model')
parser.add_argument('--results-dir', type=str, default='results', help='results dir')
parser.add_argument('--load-only', default=False, action='store_true', help='show dataset loads only')
parser.add_argument('--num-cores', type=int, default=-1, help='num parallel cores')
parser.add_argument('--pipe-memory', default=False, action='store_true', help='turn on pipeline memory')
parser.add_argument('--cache-dir', type=str, default='/tmp', help='cache dir')
parser.add_argument('--verbose', type=int, default=0, help='program verbosity')
args = parser.parse_args()
if args.test_size >= 1.0: args.test_size = int(args.test_size)
if args.scv_size >= 1.0: args.scv_size = int(args.scv_size)

base = importr('base')
biobase = importr('Biobase')
base.source('config.R')
dataset_names = list(robjects.globalenv['dataset_names'])
data_types = list(robjects.globalenv['data_types'])
norm_methods = list(robjects.globalenv['norm_methods'])
feat_types = list(robjects.globalenv['feat_types'])
prep_methods = list(robjects.globalenv['prep_methods'])
bc_methods = list(robjects.globalenv['bc_methods'])
filt_types = list(robjects.globalenv['filt_types'])
base.source('functions.R')
r_eset_class_labels = robjects.globalenv['esetClassLabels']
r_eset_feature_annots = robjects.globalenv['esetFeatureAnnots']
r_data_nzero_col_idxs = robjects.globalenv['dataNonZeroColIdxs']
r_data_nzsd_col_idxs = robjects.globalenv['dataNonZeroSdColIdxs']
r_data_nzvr_col_idxs = robjects.globalenv['dataNonZeroVarColIdxs']
r_data_corr_col_idxs = robjects.globalenv['dataCorrColIdxs']
r_limma_feature_score = robjects.globalenv['limmaFeatureScore']
r_limma_pkm_feature_score = robjects.globalenv['limmaPkmFeatureScore']
numpy2ri.activate()

if args.pipe_memory:
    cachedir = mkdtemp(dir=args.cache_dir)
    memory = Memory(cachedir=cachedir, verbose=0)
else:
    memory = None

# custom mixin and class for caching pipeline nested estimator fits
class CachedFitMixin:
    def fit(self, *args, **kwargs):
        fit = memory.cache(super(CachedFitMixin, self).fit)
        cached_self = fit(*args, **kwargs)
        vars(self).update(vars(cached_self))
        return self

class CachedLinearSVC(CachedFitMixin, LinearSVC):
    pass

class CachedExtraTreesClassifier(CachedFitMixin, ExtraTreesClassifier):
    pass

class CachedGradientBoostingClassifier(CachedFitMixin, GradientBoostingClassifier):
    pass

# limma feature selection scoring function
def limma(X, y):
    f, pv = r_limma_feature_score(X, y)
    return np.array(f), np.array(pv)

def limma_pkm(X, y):
    f, pv = r_limma_pkm_feature_score(X, y)
    return np.array(f), np.array(pv)

# parallel pipeline fit function
def fit_pipeline(params, pipeline_order, X_tr, y_tr):
    pipe_steps = sorted([
        (k, v) for k, v in params.items() if k in pipeline_order
    ], key=lambda s: pipeline_order.index(s[0]))
    pipe = Pipeline(pipe_steps, memory=memory)
    pipe.set_params(**{ k: v for k, v in params.items() if '__' in k })
    pipe.fit(X_tr, y_tr)
    if args.scv_verbose == 0: print('.', end='', flush=True)
    return pipe

# cached functions
if args.pipe_memory:
    limma_score_func = memory.cache(limma)
    limma_pkm_score_func = memory.cache(limma_pkm)
    chi2_func = memory.cache(chi2)
    f_classif_func = memory.cache(f_classif)
    mi_classif_func = memory.cache(mutual_info_classif)
    fs_svm_estimator = CachedLinearSVC()
    fs_ext_estimator = CachedExtraTreesClassifier()
    fs_grb_estimator = CachedGradientBoostingClassifier()
    sfm_svm_estimator = CachedLinearSVC(penalty='l1', dual=False)
else:
    limma_score_func = limma
    limma_pkm_score_func = limma_pkm
    chi2_func = chi2
    f_classif_func = f_classif
    mi_classif_func = mutual_info_classif
    fs_svm_estimator = LinearSVC()
    fs_ext_estimator = ExtraTreesClassifier()
    fs_grb_estimator = GradientBoostingClassifier()
    sfm_svm_estimator = LinearSVC(penalty='l1', dual=False)

# bcr performance metric scoring function
def bcr_score(y_true, y_pred):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    mes1 = (tp + fn)
    mes2 = (tn + fp)
    # if only one class
    if mes2 == 0:
        return tp / mes1
    elif mes1 == 0:
        return tn / mes2
    else:
        return (tp / mes1 + tn / mes2) / 2

scv_scoring = { 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }

# specify elements in sort order (needed by code dealing with gridsearch cv_results)
if args.slr_mms_fr_min and args.slr_mms_fr_max:
    SLR_MMS_FR = list(zip(args.slr_mms_fr_min, args.slr_mms_fr_max))
else:
    SLR_MMS_FR = [(0,1)]
if args.fs_vrt_thres:
    FS_VRT_THRES = sorted(args.fs_vrt_thres)
else:
    FS_VRT_THRES = 0.
if args.fs_skb_k:
    FS_SKB_K = sorted(args.fs_skb_k)
else:
    FS_SKB_K = list(range(1, args.fs_skb_k_max + 1, 1))
if args.fs_sfp_p:
    FS_SFP_P = sorted(args.fs_sfp_p)
else:
    FS_SFP_P = [ 1e-2, 5e-2 ]
if args.fs_sfm_svm_thres:
    FS_SFM_SVM_THRES = sorted(args.fs_sfm_svm_thres)
else:
    FS_SFM_SVM_THRES = np.logspace(-11, -5, 7)
if args.fs_sfm_svm_c:
    FS_SFM_SVM_C = sorted(args.fs_sfm_svm_c)
else:
    FS_SFM_SVM_C = np.logspace(-2, 5, 8)
if args.fs_sfm_svm_cw:
    FS_SFM_SVM_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_sfm_svm_cw],
        key=lambda x: (x is not None, x)
    )
else:
    FS_SFM_SVM_CW = [None, 'balanced']
if args.fs_sfm_ext_thres:
    FS_SFM_EXT_THRES = sorted(args.fs_sfm_ext_thres)
else:
    FS_SFM_EXT_THRES = np.logspace(-11, -5, 7)
if args.fs_sfm_ext_e:
    FS_SFM_EXT_E = sorted(args.fs_sfm_ext_e)
else:
    FS_SFM_EXT_E = list(range(5, args.fs_sfm_ext_e_max + 1, 5))
if args.fs_sfm_ext_d:
    FS_SFM_EXT_D = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_sfm_ext_d],
        key=lambda x: (x is not None, x)
    )
else:
    FS_SFM_EXT_D = [None] + list(range(1, args.fs_sfm_ext_d_max + 1, 1))
if args.fs_sfm_ext_f:
    FS_SFM_EXT_F = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_sfm_ext_f],
        key=lambda x: (x is not None, x)
    )
else:
    FS_SFM_EXT_F = [None, 'auto', 'log2', 'sqrt']
if args.fs_sfm_ext_cw:
    FS_SFM_EXT_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_sfm_ext_cw],
        key=lambda x: (x is not None, x)
    )
else:
    FS_SFM_EXT_CW = [None, 'balanced', 'balanced_subsample']
if args.fs_sfm_grb_e:
    FS_SFM_GRB_E = sorted(args.fs_sfm_grb_e)
else:
    FS_SFM_GRB_E = list(range(20, args.fs_sfm_grb_e_max + 1, 20))
if args.fs_sfm_grb_d:
    FS_SFM_GRB_D = sorted(args.fs_sfm_grb_d)
else:
    FS_SFM_GRB_D = list(range(1, args.fs_sfm_grb_d_max + 1, 1))
if args.fs_sfm_grb_f:
    FS_SFM_GRB_F = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_sfm_grb_f],
        key=lambda x: (x is not None, x)
    )
else:
    FS_SFM_GRB_F = [None, 'auto', 'log2', 'sqrt']
if args.fs_rfe_svm_c:
    FS_RFE_SVM_C = sorted(args.fs_rfe_svm_c)
else:
    FS_RFE_SVM_C = np.logspace(-7, 3, 11)
if args.fs_rfe_svm_cw:
    FS_RFE_SVM_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_rfe_svm_cw],
        key=lambda x: (x is not None, x)
    )
else:
    FS_RFE_SVM_CW = [None, 'balanced']
if args.fs_rfe_ext_e:
    FS_RFE_EXT_E = sorted(args.fs_rfe_ext_e)
else:
    FS_RFE_EXT_E = list(range(5, args.fs_rfe_ext_e_max + 1, 5))
if args.fs_rfe_ext_d:
    FS_RFE_EXT_D = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_rfe_ext_d],
        key=lambda x: (x is not None, x)
    )
else:
    FS_RFE_EXT_D = [None] + list(range(1, args.fs_rfe_ext_d_max + 1, 1))
if args.fs_rfe_ext_f:
    FS_RFE_EXT_F = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_rfe_ext_f],
        key=lambda x: (x is not None, x)
    )
else:
    FS_RFE_EXT_F = [None, 'auto', 'log2', 'sqrt']
if args.fs_rfe_ext_cw:
    FS_RFE_EXT_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_rfe_ext_cw],
        key=lambda x: (x is not None, x)
    )
else:
    FS_RFE_EXT_CW = [None, 'balanced', 'balanced_subsample']
if args.fs_rfe_grb_e:
    FS_RFE_GRB_E = sorted(args.fs_rfe_grb_e)
else:
    FS_RFE_GRB_E = list(range(20, args.fs_rfe_grb_e_max + 1, 20))
if args.fs_rfe_grb_d:
    FS_RFE_GRB_D = sorted(args.fs_rfe_grb_d)
else:
    FS_RFE_GRB_D = list(range(1, args.fs_rfe_grb_d_max + 1, 1))
if args.fs_rfe_grb_f:
    FS_RFE_GRB_F = sorted(
        [None if a in ('None', 'none') else a for a in args.fs_rfe_grb_f],
        key=lambda x: (x is not None, x)
    )
else:
    FS_RFE_GRB_F = [None, 'auto', 'log2', 'sqrt']
if args.fs_rfe_step:
    FS_RFE_STEP = sorted(args.fs_rfe_step)
else:
    FS_RFE_STEP = [1]
if args.fs_rlf_n:
    FS_RLF_N = sorted(args.fs_rlf_n)
else:
    FS_RLF_N = list(range(1, args.fs_rlf_n_max + 1, 1))
if args.fs_rlf_s:
    FS_RLF_S = sorted(args.fs_rlf_s)
else:
    FS_RLF_S = list(range(1, args.fs_rlf_s_max + 1, 1))
if args.clf_svm_c:
    CLF_SVM_C = sorted(args.clf_svm_c)
else:
    CLF_SVM_C = np.logspace(-7, 3, 11)
if args.clf_svm_cw:
    CLF_SVM_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_svm_cw],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_SVM_CW = [None, 'balanced']
if args.clf_svm_kern:
    CLF_SVM_KERN = sorted(args.clf_svm_kern)
else:
    CLF_SVM_KERN = ['linear', 'poly', 'rbf', 'sigmoid']
if args.clf_svm_deg:
    CLF_SVM_DEG = sorted(args.clf_svm_deg)
else:
    CLF_SVM_DEG = [2, 3, 4, 5]
if args.clf_svm_g:
    CLF_SVM_G = sorted(args.clf_svm_g)
else:
    CLF_SVM_G = ['auto'] + list(np.logspace(-7, 3, 11))
if args.clf_knn_k:
    CLF_KNN_K = sorted(args.clf_knn_k)
else:
    CLF_KNN_K = list(range(1, args.clf_knn_k_max + 1, 1))
if args.clf_knn_w:
    CLF_KNN_W = sorted(args.clf_knn_w)
else:
    CLF_KNN_W = ['distance', 'uniform']
if args.clf_dt_d:
    CLF_DT_D = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_dt_d],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_DT_D = [None] + list(range(1, args.clf_dt_d_max + 1, 1))
if args.clf_dt_f:
    CLF_DT_F = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_dt_f],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_DT_F = [None, 'auto', 'log2', 'sqrt']
if args.clf_dt_cw:
    CLF_DT_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_dt_cw],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_DT_CW = [None, 'balanced']
if args.clf_rf_e:
    CLF_RF_E = sorted(args.clf_rf_e)
else:
    CLF_RF_E = list(range(5, args.clf_rf_e_max + 1, 5))
if args.clf_rf_d:
    CLF_RF_D = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_rf_d],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_RF_D = [None] + list(range(1, args.clf_rf_d_max + 1, 1))
if args.clf_rf_f:
    CLF_RF_F = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_rf_f],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_RF_F = [None, 'auto', 'log2', 'sqrt']
if args.clf_rf_cw:
    CLF_RF_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_rf_cw],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_RF_CW = [None, 'balanced']
if args.clf_ext_e:
    CLF_EXT_E = sorted(args.clf_ext_e)
else:
    CLF_EXT_E = list(range(5, args.clf_ext_e_max + 1, 5))
if args.clf_ext_d:
    CLF_EXT_D = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_ext_d],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_EXT_D = [None] + list(range(1, args.clf_ext_d_max + 1, 1))
if args.clf_ext_f:
    CLF_EXT_F = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_ext_f],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_EXT_F = [None, 'auto', 'log2', 'sqrt']
if args.clf_ext_cw:
    CLF_EXT_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_ext_cw],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_EXT_CW = [None, 'balanced', 'balanced_subsample']
if args.clf_ada_e:
    CLF_ADA_E = sorted(args.clf_ada_e)
else:
    CLF_ADA_E = list(range(20, args.clf_ada_e_max + 1, 20))
if args.clf_ada_lgr_c:
    CLF_ADA_LGR_C = sorted(args.clf_ada_lgr_c)
else:
    CLF_ADA_LGR_C = np.logspace(-7, 3, 11)
if args.clf_ada_lgr_cw:
    CLF_ADA_LGR_CW = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_ada_lgr_cw],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_ADA_LGR_CW = [None, 'balanced']
if args.clf_grb_e:
    CLF_GRB_E = sorted(args.clf_grb_e)
else:
    CLF_GRB_E = list(range(20, args.clf_grb_e_max + 1, 20))
if args.clf_grb_d:
    CLF_GRB_D = sorted(args.clf_grb_d)
else:
    CLF_GRB_D = list(range(1, args.clf_grb_d_max + 1, 1))
if args.clf_grb_f:
    CLF_GRB_F = sorted(
        [None if a in ('None', 'none') else a for a in args.clf_grb_f],
        key=lambda x: (x is not None, x)
    )
else:
    CLF_GRB_F = [None, 'auto', 'log2', 'sqrt']
if args.clf_mlp_hls:
    CLF_MLP_HLS = tuple(args.clf_mlp_hls)
else:
    CLF_MLP_HLS = (10,)
if args.clf_mlp_act:
    CLF_MLP_ACT = sorted(args.clf_mlp_act)
else:
    CLF_MLP_ACT = ['identity', 'logistic', 'relu', 'tanh']
if args.clf_mlp_slvr:
    CLF_MLP_SLVR = sorted(args.clf_mlp_slvr)
else:
    CLF_MLP_SLVR = ['adam', 'lbfgs', 'sgd']
if args.clf_mlp_a:
    CLF_MLP_A = sorted(args.clf_mlp_a)
else:
    CLF_MLP_A = np.logspace(-7, 3, 11)
if args.clf_mlp_lr:
    CLF_MLP_LR = sorted(args.clf_mlp_lr)
else:
    CLF_MLP_LR = ['adaptive', 'constant', 'invscaling']

pipeline_order = [
    'fs1',
    'slr',
    'fs2',
    'clf',
]
pipelines = {
    'slr': {
        'None': {
            'steps': [
                ('slr', None),
            ],
            'param_grid': [
                { },
            ],
        },
        'MinMaxScaler': {
            'steps': [
                ('slr', MinMaxScaler()),
            ],
            'param_grid': [
                {
                    'slr__feature_range': SLR_MMS_FR,
                },
            ],
        },
        'StandardScaler': {
            'steps': [
                ('slr', StandardScaler()),
            ],
            'param_grid': [
                { },
            ],
        },
        'RobustScaler': {
            'steps': [
                ('slr', RobustScaler()),
            ],
            'param_grid': [
                { },
            ],
        },
    },
    'fs': {
        'None': {
            'steps': [

            ],
            'param_grid': [
                { },
            ],
        },
        'ANOVA-KBest': {
            'steps': [
                ('fs1', SelectKBest(f_classif_func)),
            ],
            'param_grid': [
                {
                    'fs1__k': FS_SKB_K,
                },
            ],
        },
        'Chi2-KBest': {
            'steps': [
                ('fs1', SelectKBest(chi2_func)),
            ],
            'param_grid': [
                {
                    'fs1__k': FS_SKB_K,
                },
            ],
        },
        'Limma-KBest': {
            'steps': [
                ('fs1', SelectKBest()),
            ],
            'param_grid': [
                {
                    'fs1__k': FS_SKB_K,
                },
            ],
        },
        'VarianceThreshold': {
            'steps': [
                ('fs2', VarianceThreshold()),
            ],
            'param_grid': [
                {
                    'fs2__threshold': FS_VRT_THRES,
                },
            ],
        },
        'MI-KBest': {
            'steps': [
                ('fs2', SelectKBest(mi_classif_func)),
            ],
            'param_grid': [
                {
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'SVM-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(sfm_svm_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__C': FS_SFM_SVM_C,
                    'fs2__estimator__class_weight': FS_SFM_SVM_CW,
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'EXT-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(fs_ext_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_SFM_EXT_E,
                    'fs2__estimator__max_depth': FS_SFM_EXT_D,
                    'fs2__estimator__max_features': FS_SFM_EXT_F,
                    'fs2__estimator__class_weight': FS_SFM_EXT_CW,
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'GRB-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(fs_grb_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_SFM_GRB_E,
                    'fs2__estimator__max_depth': FS_SFM_GRB_D,
                    'fs2__estimator__max_features': FS_SFM_GRB_F,
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'SVM-RFE': {
            'steps': [
                ('fs2', RFE(fs_svm_estimator, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__C': FS_RFE_SVM_C,
                    'fs2__estimator__class_weight': FS_RFE_SVM_CW,
                    'fs2__step': FS_RFE_STEP,
                    'fs2__n_features_to_select': FS_SKB_K,
                },
            ],
        },
        'EXT-RFE': {
            'steps': [
                ('fs2', RFE(fs_ext_estimator, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_RFE_EXT_E,
                    'fs2__estimator__max_depth': FS_RFE_EXT_D,
                    'fs2__estimator__max_features': FS_RFE_EXT_F,
                    'fs2__estimator__class_weight': FS_RFE_EXT_CW,
                    'fs2__step': FS_RFE_STEP,
                    'fs2__n_features_to_select': FS_SKB_K,
                },
            ],
        },
        'GRB-RFE': {
            'steps': [
                ('fs2', RFE(fs_grb_estimator, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_RFE_GRB_E,
                    'fs2__estimator__max_depth': FS_RFE_GRB_D,
                    'fs2__estimator__max_features': FS_RFE_GRB_F,
                    'fs2__step': FS_RFE_STEP,
                    'fs2__n_features_to_select': FS_SKB_K,
                },
            ],
        },
        'FCBF': {
            'steps': [
                ('fs2', FCBF(memory=memory)),
            ],
            'param_grid': [
                {
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'ReliefF': {
            'steps': [
                ('fs2', ReliefF(memory=memory)),
            ],
            'param_grid': [
                {
                    'fs2__k': FS_SKB_K,
                    'fs2__n_neighbors': FS_RLF_N,
                    'fs2__sample_size': FS_RLF_S,
                },
            ],
        },
        'CFS': {
            'steps': [
                ('fs2', CFS()),
            ],
            'param_grid': [
                { },
            ],
        },
    },
    'clf': {
        'LinearSVM': {
            'steps': [
                ('clf', LinearSVC()),
            ],
            'param_grid': [
                {
                    'clf__C': CLF_SVM_C,
                    'clf__class_weight': CLF_SVM_CW,
                },
            ],
        },
        'KernelSVM': {
            'steps': [
                ('clf', SVC(cache_size=args.clf_svm_cache)),
            ],
            'param_grid': [
                {
                    'clf__C': CLF_SVM_C,
                    'clf__class_weight': CLF_SVM_CW,
                    'clf__kernel': CLF_SVM_KERN,
                    'clf__degree': CLF_SVM_DEG,
                    'clf__gamma': CLF_SVM_G,
                },
            ],
        },
        'kNN': {
            'steps': [
                ('clf', KNeighborsClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_neighbors': CLF_KNN_K,
                    'clf__weights': CLF_KNN_W,
                },
            ],
        },
        'DT': {
            'steps': [
                ('clf', DecisionTreeClassifier()),
            ],
            'param_grid': [
                {
                    'clf__max_depth': CLF_DT_D,
                    'clf__max_features': CLF_DT_F,
                    'clf__class_weight': CLF_DT_CW,
                },
            ],
        },
        'RandomForest': {
            'steps': [
                ('clf', RandomForestClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_RF_E,
                    'clf__max_depth': CLF_RF_D,
                    'clf__max_features': CLF_RF_F,
                    'clf__class_weight': CLF_RF_CW,
                },
            ],
        },
        'ExtraTrees': {
            'steps': [
                ('clf', ExtraTreesClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_EXT_E,
                    'clf__max_depth': CLF_EXT_D,
                    'clf__max_features': CLF_EXT_F,
                    'clf__class_weight': CLF_EXT_CW,
                },
            ],
        },
        'AdaBoost': {
            'steps': [
                ('clf', AdaBoostClassifier(LogisticRegression())),
            ],
            'param_grid': [
                {
                    'clf__base_estimator__C': CLF_ADA_LGR_C,
                    'clf__base_estimator__class_weight': CLF_ADA_LGR_CW,
                    'clf__n_estimators': CLF_ADA_E,
                },
            ],
        },
        'GradientBoost': {
            'steps': [
                ('clf', GradientBoostingClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_GRB_E,
                    'clf__max_depth': CLF_GRB_D,
                    'clf__max_features': CLF_GRB_F,
                },
            ],
        },
        'GaussianNB': {
            'steps': [
                ('clf', GaussianNB()),
            ],
            'param_grid': [
                { },
            ],
        },
        'GaussianProcess': {
            'steps': [
                ('clf', GaussianProcessClassifier(1.0 * RBF(1.0))),
            ],
            'param_grid': [
                { },
            ],
        },
        'LDA': {
            'steps': [
                ('clf', LinearDiscriminantAnalysis()),
            ],
            'param_grid': [
                { },
            ],
        },
        'QDA': {
            'steps': [
                ('clf', QuadraticDiscriminantAnalysis()),
            ],
            'param_grid': [
                { },
            ],
        },
        'MLP': {
            'steps': [
                ('clf', MLPClassifier()),
            ],
            'param_grid': [
                {
                    'clf__hidden_layer_sizes': CLF_MLP_HLS,
                    'clf__activation': CLF_MLP_ACT,
                    'clf__solver': CLF_MLP_SLVR,
                    'clf__alpha': CLF_MLP_A,
                    'clf__learning_rate': CLF_MLP_LR,
                },
            ],
        },
    },
}
params_feature_select = [
    'fs1__k',
    'fs2__k',
    'fs2__n_features_to_select',
]
params_num_xticks = [
    'fs1__k',
    'fs2__k',
    'fs2__estimator__n_estimators',
    'fs2__n_neighbors',
    'fs2__sample_size',
    'fs2__n_features_to_select',
    'clf__degree',
    'clf__n_neighbors',
    'clf__n_estimators',
]
params_fixed_xticks = [
    'fs1__alpha',
    'fs2__estimator__C',
    'fs2__estimator__class_weight',
    'fs2__estimator__max_depth',
    'fs2__estimator__max_features',
    'fs2__threshold',
    'clf__C',
    'clf__class_weight',
    'clf__kernel',
    'clf__gamma',
    'clf__weights',
    'clf__max_depth',
    'clf__base_estimator__C',
    'clf__base_estimator__class_weight',
    'clf__max_features',
]

# analyses
if args.analysis == 1:
    prep_steps = []
    if args.data_type and args.data_type[0] not in ('None', 'none'):
        data_type = [x for x in data_types if x in args.data_type][0]
        prep_steps.append(data_type)
    else:
        data_type = None
    if args.norm_meth and args.norm_meth[0] not in ('None', 'none'):
        norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
        prep_steps.append(norm_meth)
    else:
        norm_meth = None
    if args.feat_type and args.feat_type[0] not in ('None', 'none'):
        feat_type = [x for x in feat_types if x in args.feat_type][0]
        prep_steps.append(feat_type)
    else:
        feat_type = None
    if args.prep_meth and args.prep_meth[0] not in ('None', 'none'):
        prep_meth = [x for x in prep_methods if x in args.prep_meth][0]
        prep_steps.append(prep_meth)
    else:
        prep_meth = None
    if args.bc_meth and args.bc_meth[0] not in ('None', 'none'):
        bc_meth = [x for x in bc_methods if x in args.bc_meth][0]
        prep_steps.append(bc_meth)
    else:
        bc_meth = None
    if args.filt_type and args.filt_type[0] not in ('None', 'none'):
        filt_type = [x for x in filt_types if x in args.filt_type][0]
        prep_steps.append(filt_type)
    else:
        filt_type = None
    args.fs_meth = args.fs_meth[0]
    args.slr_meth = args.slr_meth[0]
    args.clf_meth = args.clf_meth[0]
    args.dataset_tr = natsorted(args.dataset_tr)
    if len(args.dataset_tr) > 1 or bc_meth:
        dataset_name = '_'.join(args.dataset_tr + prep_steps + ['tr'])
    else:
        dataset_name = '_'.join(args.dataset_tr + prep_steps)
    eset_name = 'eset_' + dataset_name
    eset_file = 'data/' + eset_name + '.Rda'
    if path.isfile(eset_file):
        base.load('data/' + eset_name + '.Rda')
    else:
        exit('File does not exist or invalid: ' + eset_file)
    eset = robjects.globalenv[eset_name]
    X = np.array(base.t(biobase.exprs(eset)), dtype=float)
    y = np.array(r_eset_class_labels(eset), dtype=int)
    if args.filt_zero_feats:
        nzero_feature_idxs = np.array(r_data_nzero_col_idxs(X), dtype=int)
        X = X[:, nzero_feature_idxs]
    if args.filt_zero_samples:
        nzero_sample_idxs = np.array(r_data_nzero_col_idxs(X.T), dtype=int)
        X = X[nzero_sample_idxs, :]
        y = y[nzero_sample_idxs]
    if args.filt_zsdv_feats:
        nzsd_feature_idxs = np.array(r_data_nzsd_col_idxs(X), dtype=int)
        X = X[:, nzsd_feature_idxs]
    if args.filt_zsdv_samples:
        nzsd_sample_idxs = np.array(r_data_nzsd_col_idxs(X.T), dtype=int)
        X = X[nzsd_sample_idxs, :]
        y = y[nzsd_sample_idxs]
    if args.filt_nzvr_feats:
        nzvr_feature_idxs = np.array(r_data_nzvr_col_idxs(
            X, freqCut=args.filt_nzvr_feat_freq_cut, uniqueCut=args.filt_nzvr_feat_uniq_cut
        ), dtype=int)
        X = X[:, nzvr_feature_idxs]
    if args.filt_nzvr_samples:
        nzvr_sample_idxs = np.array(r_data_nzvr_col_idxs(
            X.T, freqCut=args.filt_nzvr_sample_freq_cut, uniqueCut=args.filt_nzvr_sample_uniq_cut
        ), dtype=int)
        X = X[nzvr_sample_idxs, :]
        y = y[nzvr_sample_idxs]
    if args.filt_ncor_feats:
        corr_feature_idxs = np.array(
            r_data_corr_col_idxs(X, cutoff=args.filt_ncor_feat_cut), dtype=int
        )
        X = X[:, corr_feature_idxs]
    if args.filt_ncor_samples:
        corr_sample_idxs = np.array(
            r_data_corr_col_idxs(X.T, cutoff=args.filt_ncor_sample_cut), dtype=int
        )
        X = X[corr_sample_idxs, :]
        y = y[corr_sample_idxs]
    pipe = Pipeline(sorted(
        pipelines['slr'][args.slr_meth]['steps'] +
        pipelines['fs'][args.fs_meth]['steps'] +
        pipelines['clf'][args.clf_meth]['steps'],
        key=lambda s: pipeline_order.index(s[0])
    ), memory=memory)
    param_grid = {
        **pipelines['slr'][args.slr_meth]['param_grid'][0],
        **pipelines['fs'][args.fs_meth]['param_grid'][0],
        **pipelines['clf'][args.clf_meth]['param_grid'][0],
    }
    if args.fs_meth == 'Limma-KBest':
        if norm_meth and norm_meth == 'pkm':
            pipe.set_params(fs1__score_func=limma_pkm_score_func)
        else:
            pipe.set_params(fs1__score_func=limma_score_func)
    for param in param_grid:
        if param in params_feature_select:
            param_grid[param] = list(filter(
                lambda x: x <= min(
                    X.shape[1],
                    args.fs_skb_k_max if args.fs_skb_lim_off else y.shape[0]
                ),
                param_grid[param]
            ))
    if args.scv_type == 'grid':
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring=scv_scoring, refit=args.scv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
        )
    elif args.scv_type == 'rand':
        search = RandomizedSearchCV(
            pipe, param_distributions=param_grid, scoring=scv_scoring, refit=args.scv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
            n_iter=args.scv_n_iter,
        )
    if args.verbose > 0:
        print('Pipeline:')
        pprint(vars(pipe))
        print('Param grid:')
        pprint(param_grid)
    print('Dataset:', dataset_name, X.shape, y.shape)
    split_num = 1
    split_results = []
    param_scores_cv = {}
    sss = StratifiedShuffleSplit(n_splits=args.test_splits, test_size=args.test_size)
    for tr_idxs, te_idxs in sss.split(X, y):
        search.fit(X[tr_idxs], y[tr_idxs])
        feature_idxs = np.arange(X[tr_idxs].shape[1])
        for step in search.best_estimator_.named_steps:
            if hasattr(search.best_estimator_.named_steps[step], 'get_support'):
                feature_idxs = feature_idxs[search.best_estimator_.named_steps[step].get_support(indices=True)]
        feature_names = np.array(biobase.featureNames(eset), dtype=str)[feature_idxs]
        weights = np.array([], dtype=float)
        if hasattr(search.best_estimator_.named_steps['clf'], 'coef_'):
            weights = np.square(search.best_estimator_.named_steps['clf'].coef_[0])
        elif hasattr(search.best_estimator_.named_steps['clf'], 'feature_importances_'):
            weights = search.best_estimator_.named_steps['clf'].feature_importances_
        elif ('fs2' in search.best_estimator_.named_steps):
            if (hasattr(search.best_estimator_.named_steps['fs2'], 'estimator_') and
                hasattr(search.best_estimator_.named_steps['fs2'].estimator_, 'coef_')):
                weights = np.square(search.best_estimator_.named_steps['fs2'].estimator_.coef_[0])
            elif hasattr(search.best_estimator_.named_steps['fs2'], 'scores_'):
                weights = search.best_estimator_.named_steps['fs2'].scores_
            elif hasattr(search.best_estimator_.named_steps['fs2'], 'feature_importances_'):
                weights = search.best_estimator_.named_steps['fs2'].feature_importances_
        roc_auc_cv = search.cv_results_['mean_test_roc_auc'][search.best_index_]
        bcr_cv = search.cv_results_['mean_test_bcr'][search.best_index_]
        if hasattr(search, 'decision_function'):
            y_score = search.decision_function(X[te_idxs])
        else:
            y_score = search.predict_proba(X[te_idxs])[:,1]
        roc_auc_te = roc_auc_score(y[te_idxs], y_score)
        fpr, tpr, thres = roc_curve(y[te_idxs], y_score, pos_label=1)
        y_pred = search.predict(X[te_idxs])
        bcr_te = bcr_score(y[te_idxs], y_pred)
        print(
            'Dataset:', dataset_name,
            ' Split: %2s' % split_num,
            ' ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
            ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
            ' Features: %3s' % feature_idxs.size,
            ' Params:', search.best_params_,
        )
        if args.verbose > 0:
            if weights.size > 0:
                print('Feature Rankings:')
                if args.show_annots:
                    feature_ranks = sorted(
                        zip(feature_idxs, feature_names, weights, r_eset_feature_annots(
                            eset,
                            annots=robjects.StrVector(args.show_annots),
                            features=(feature_idxs + 1)
                        )),
                        key=itemgetter(2), reverse=True
                    )
                    for _, feature_name, weight, annot in feature_ranks:
                        print(feature_name, '\t', weight, '\t', annot)
                else:
                    feature_ranks = sorted(
                        zip(feature_idxs, feature_names, weights),
                        key=itemgetter(2), reverse=True
                    )
                    for _, feature_name, weight in feature_ranks:
                        print(feature_name, '\t', weight)
            else:
                print('Features:')
                if args.show_annots:
                    feature_ranks = sorted(
                        zip(feature_idxs, feature_names, r_eset_feature_annots(
                            eset,
                            annots=robjects.StrVector(args.show_annots),
                            features=(feature_idxs + 1)
                        )),
                        key=itemgetter(1)
                    )
                    for _, feature_name, annot in feature_ranks:
                        print(feature_name, '\t', annot)
                else:
                    feature_ranks = sorted(
                        zip(feature_idxs, feature_names),
                        key=itemgetter(1)
                    )
                    for _, feature_name in feature_ranks: print(feature_name)
        for param, param_values in param_grid.items():
            if len(param_values) == 1: continue
            new_shape = (
                len(param_values),
                int(len(search.cv_results_['params']) / len(param_values))
            )
            param_values_cv = np.ma.getdata(search.cv_results_['param_%s' % param])
            param_values_cv_sorted_idxs = np.where(
                np.array(param_values).reshape(len(param_values), 1) == param_values_cv
            )[1]
            if param not in param_scores_cv: param_scores_cv[param] = {}
            for metric in scv_scoring.keys():
                if args.scv_h_plt_meth == 'best':
                    mean_scores_cv = np.max(np.transpose(np.reshape(
                        search.cv_results_[
                            'mean_test_%s' % metric
                        ][param_values_cv_sorted_idxs],
                        new_shape
                    )), axis=0)
                    if metric in param_scores_cv[param]:
                        param_scores_cv[param][metric] = np.vstack(
                            (param_scores_cv[param][metric], mean_scores_cv)
                        )
                    else:
                        param_scores_cv[param][metric] = mean_scores_cv
                elif args.scv_h_plt_meth == 'all':
                    for split_idx in range(search.n_splits_):
                        split_scores_cv = np.transpose(np.reshape(
                            search.cv_results_[
                                'split%d_test_%s' % (split_idx, metric)
                            ][param_values_cv_sorted_idxs],
                            new_shape
                        ))
                        if metric in param_scores_cv[param]:
                            param_scores_cv[param][metric] = np.vstack(
                                (param_scores_cv[param][metric], split_scores_cv)
                            )
                        else:
                            param_scores_cv[param][metric] = split_scores_cv
        split_results.append({
            'search': search,
            'feature_idxs': feature_idxs,
            'feature_names': feature_names,
            'fprs': fpr,
            'tprs': tpr,
            'thres': thres,
            'weights': weights,
            'y_score': y_score,
            'roc_auc_cv': roc_auc_cv,
            'roc_auc_te': roc_auc_te,
            'bcr_cv': bcr_cv,
            'bcr_te': bcr_te,
        })
        split_num += 1
        # flush cache with each combo run (grows too big if not)
        if args.pipe_memory: memory.clear(warn=False)
    # plot grid search parameters vs cv perf metrics
    sns.set_palette(sns.color_palette('hls', len(scv_scoring)))
    for param_idx, param in enumerate(param_scores_cv):
        mean_roc_aucs_cv = np.mean(param_scores_cv[param]['roc_auc'], axis=0)
        mean_bcrs_cv = np.mean(param_scores_cv[param]['bcr'], axis=0)
        std_roc_aucs_cv = np.std(param_scores_cv[param]['roc_auc'], axis=0)
        std_bcrs_cv = np.std(param_scores_cv[param]['bcr'], axis=0)
        plt.figure('Figure ' + str(args.analysis) + '-' + str(param_idx + 1))
        plt.rcParams['font.size'] = 14
        if param in params_num_xticks:
            x_axis = param_grid[param]
            plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
            plt.xticks(x_axis)
        elif param in params_fixed_xticks:
            x_axis = range(len(param_grid[param]))
            plt.xticks(x_axis, param_grid[param])
        plt.title(
            dataset_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
            'Effect of ' + param + ' on CV Performance Metrics'
        )
        plt.xlabel(param)
        plt.ylabel('CV Score')
        plt.plot(
            x_axis,
            mean_roc_aucs_cv,
            lw=2, alpha=0.8, label='Mean ROC AUC'
        )
        plt.fill_between(
            x_axis,
            [m - s for m, s in zip(mean_roc_aucs_cv, std_roc_aucs_cv)],
            [m + s for m, s in zip(mean_roc_aucs_cv, std_roc_aucs_cv)],
            color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
        )
        plt.plot(
            x_axis,
            mean_bcrs_cv,
            lw=2, alpha=0.8, label='Mean BCR'
        )
        plt.fill_between(
            x_axis,
            [m - s for m, s in zip(mean_bcrs_cv, std_bcrs_cv)],
            [m + s for m, s in zip(mean_bcrs_cv, std_bcrs_cv)],
            color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
        )
        plt.legend(loc='lower right', fontsize='small')
        plt.grid('on')
    # plot roc curve
    sns.set_palette(sns.color_palette('hls', 2))
    plt.figure('Figure ' + str(args.analysis) + '-' + str(len(param_scores_cv) + 1))
    plt.rcParams['font.size'] = 14
    plt.title(
        dataset_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'ROC Curve'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    tprs, roc_aucs_cv, roc_aucs_te, bcrs_cv, bcrs_te, num_features = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    for split_idx, split_result in enumerate(split_results):
        roc_aucs_cv.append(split_result['roc_auc_cv'])
        roc_aucs_te.append(split_result['roc_auc_te'])
        bcrs_cv.append(split_result['bcr_cv'])
        bcrs_te.append(split_result['bcr_te'])
        num_features.append(split_result['feature_idxs'].size)
        tprs.append(np.interp(mean_fpr, split_result['fprs'], split_result['tprs']))
        tprs[-1][0] = 0.0
        plt.plot(
            split_result['fprs'], split_result['tprs'], color='darkgrey', lw=1, alpha=0.2,
            # label='ROC split %d (AUC = %0.4f)' % (split_idx + 1, split_result['roc_auc_te']),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs_te)
    std_roc_auc = np.std(roc_aucs_te)
    mean_num_features = np.mean(num_features)
    std_num_features = np.std(num_features)
    plt.plot(
        mean_fpr, mean_tpr, lw=3, alpha=0.8,
        label=r'Test Mean ROC (AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' %
        (mean_roc_auc, std_roc_auc, mean_num_features, std_num_features),
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, alpha=0.2, label='Chance')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid('off')
    print(
        'Dataset:', dataset_name,
        ' Mean ROC AUC (CV / Test): %.4f / %.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
        ' Mean BCR (CV / Test): %.4f / %.4f' % (np.mean(bcrs_cv), np.mean(bcrs_te)),
        ' Mean Features: %3d' % np.mean(num_features),
    )
    # calculate overall best ranked features
    feature_idxs = []
    for split_result in split_results: feature_idxs.extend(split_result['feature_idxs'])
    feature_idxs = sorted(list(set(feature_idxs)))
    feature_names = np.array(biobase.featureNames(eset), dtype=str)[feature_idxs]
    # print(*natsorted(feature_names), sep='\n')
    feature_mx_idx = {}
    for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
    weight_mx = np.zeros((len(feature_idxs), len(split_results)), dtype=float)
    roc_auc_mx = np.zeros((len(feature_idxs), len(split_results)), dtype=float)
    bcr_mx = np.zeros((len(feature_idxs), len(split_results)), dtype=float)
    for split_idx, split_result in enumerate(split_results):
        for idx, feature_idx in enumerate(split_result['feature_idxs']):
            weight_mx[feature_mx_idx[feature_idx]][split_idx] = split_result['weights'][idx]
            roc_auc_mx[feature_mx_idx[feature_idx]][split_idx] = split_result['roc_auc_cv']
            bcr_mx[feature_mx_idx[feature_idx]][split_idx] = split_result['bcr_cv']
    feature_mean_weights, feature_mean_roc_aucs, feature_mean_bcrs = [], [], []
    for idx in range(len(feature_idxs)):
        feature_mean_weights.append(np.mean(weight_mx[idx]))
        feature_mean_roc_aucs.append(np.mean(roc_auc_mx[idx]))
        feature_mean_bcrs.append(np.mean(bcr_mx[idx]))
        # print(feature_names[idx], '\t', feature_mean_weights[idx], '\t', weight_mx[idx])
        # print(feature_names[idx], '\t', feature_mean_roc_aucs[idx], '\t', roc_auc_mx[idx])
    if args.fs_rank_meth == 'mean_weights':
        feature_ranks = feature_mean_weights
    elif args.fs_rank_meth == 'mean_roc_aucs':
        feature_ranks = feature_mean_roc_aucs
    elif args.fs_rank_meth == 'mean_bcrs':
        feature_ranks = feature_mean_bcrs
    print('Overall Feature Rankings:')
    for rank, feature in sorted(zip(feature_ranks, feature_names), reverse=True):
        print(feature, '\t', rank)
elif args.analysis == 2:
    prep_steps = []
    if args.data_type and args.data_type[0] not in ('None', 'none'):
        data_type = [x for x in data_types if x in args.data_type][0]
        prep_steps.append(data_type)
    else:
        data_type = None
    if args.norm_meth and args.norm_meth[0] not in ('None', 'none'):
        norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
        prep_steps.append(norm_meth)
    else:
        norm_meth = None
    if args.feat_type and args.feat_type[0] not in ('None', 'none'):
        feat_type = [x for x in feat_types if x in args.feat_type][0]
        prep_steps.append(feat_type)
    else:
        feat_type = None
    if args.prep_meth and args.prep_meth[0] not in ('None', 'none'):
        prep_meth = [x for x in prep_methods if x in args.prep_meth][0]
        prep_steps.append(prep_meth)
    else:
        prep_meth = None
    if args.bc_meth and args.bc_meth[0] not in ('None', 'none'):
        bc_meth = [x for x in bc_methods if x in args.bc_meth][0]
        prep_steps.append(bc_meth)
    else:
        bc_meth = None
    if args.filt_type and args.filt_type[0] not in ('None', 'none'):
        filt_type = [x for x in filt_types if x in args.filt_type][0]
        prep_steps.append(filt_type)
    else:
        filt_type = None
    args.fs_meth = args.fs_meth[0]
    args.slr_meth = args.slr_meth[0]
    args.clf_meth = args.clf_meth[0]
    args.dataset_tr = natsorted(args.dataset_tr)
    if len(args.dataset_tr) > 1 or bc_meth:
        dataset_tr_name = '_'.join(args.dataset_tr + prep_steps + ['tr'])
    else:
        dataset_tr_name = '_'.join(args.dataset_tr + prep_steps)
    eset_tr_name = 'eset_' + dataset_tr_name
    eset_tr_file = 'data/' + eset_tr_name + '.Rda'
    if path.isfile(eset_tr_file):
        base.load('data/' + eset_tr_name + '.Rda')
    else:
        exit('File does not exist or invalid: ' + eset_tr_file)
    eset_tr = robjects.globalenv[eset_tr_name]
    X_tr = np.array(base.t(biobase.exprs(eset_tr)), dtype=float)
    y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
    if args.filt_zero_feats:
        nzero_feature_idxs = np.array(r_data_nzero_col_idxs(X_tr), dtype=int)
        X_tr = X_tr[:, nzero_feature_idxs]
    if args.filt_zero_samples:
        nzero_sample_idxs = np.array(r_data_nzero_col_idxs(X_tr.T), dtype=int)
        X_tr = X_tr[nzero_sample_idxs, :]
        y_tr = y_tr[nzero_sample_idxs]
    if args.filt_zsdv_feats:
        nzsd_feature_idxs = np.array(r_data_nzsd_col_idxs(X_tr), dtype=int)
        X_tr = X_tr[:, nzsd_feature_idxs]
    if args.filt_zsdv_samples:
        nzsd_sample_idxs = np.array(r_data_nzsd_col_idxs(X_tr.T), dtype=int)
        X_tr = X_tr[nzsd_sample_idxs, :]
        y_tr = y_tr[nzsd_sample_idxs]
    if args.filt_nzvr_feats:
        nzvr_feature_idxs = np.array(r_data_nzvr_col_idxs(
            X_tr, freqCut=args.filt_nzvr_feat_freq_cut, uniqueCut=args.filt_nzvr_feat_uniq_cut
        ), dtype=int)
        X_tr = X_tr[:, nzvr_feature_idxs]
    if args.filt_nzvr_samples:
        nzvr_sample_idxs = np.array(r_data_nzvr_col_idxs(
            X_tr.T, freqCut=args.filt_nzvr_sample_freq_cut, uniqueCut=args.filt_nzvr_sample_uniq_cut
        ), dtype=int)
        X_tr = X_tr[nzvr_sample_idxs, :]
        y_tr = y_tr[nzvr_sample_idxs]
    if args.filt_ncor_feats:
        corr_feature_idxs = np.array(
            r_data_corr_col_idxs(X_tr, cutoff=args.filt_ncor_feat_cut), dtype=int
        )
        X_tr = X_tr[:, corr_feature_idxs]
    if args.filt_ncor_samples:
        corr_sample_idxs = np.array(
            r_data_corr_col_idxs(X_tr.T, cutoff=args.filt_ncor_sample_cut), dtype=int
        )
        X_tr = X_tr[corr_sample_idxs, :]
        y_tr = y_tr[corr_sample_idxs]
    pipe = Pipeline(sorted(
        pipelines['slr'][args.slr_meth]['steps'] +
        pipelines['fs'][args.fs_meth]['steps'] +
        pipelines['clf'][args.clf_meth]['steps'],
        key=lambda s: pipeline_order.index(s[0])
    ), memory=memory)
    param_grid = {
        **pipelines['slr'][args.slr_meth]['param_grid'][0],
        **pipelines['fs'][args.fs_meth]['param_grid'][0],
        **pipelines['clf'][args.clf_meth]['param_grid'][0],
    }
    if args.fs_meth == 'Limma-KBest':
        if norm_meth and norm_meth == 'pkm':
            pipe.set_params(fs1__score_func=limma_pkm_score_func)
        else:
            pipe.set_params(fs1__score_func=limma_score_func)
    for param in param_grid:
        if param in params_feature_select:
            param_grid[param] = list(filter(
                lambda x: x <= min(
                    X_tr.shape[1],
                    args.fs_skb_k_max if args.fs_skb_lim_off else y_tr.shape[0]
                ),
                param_grid[param]
            ))
    if args.scv_type == 'grid':
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring=scv_scoring, refit=args.scv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
        )
    elif args.scv_type == 'rand':
        search = RandomizedSearchCV(
            pipe, param_distributions=param_grid, scoring=scv_scoring, refit=args.scv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
            n_iter=args.scv_n_iter,
        )
    if args.verbose > 0:
        print('Pipeline:')
        pprint(vars(pipe))
        print('Param grid:')
        pprint(param_grid)
    print('Train:', dataset_tr_name, X_tr.shape, y_tr.shape)
    search.fit(X_tr, y_tr)
    if args.save_model:
        makedirs(args.results_dir, mode=0o755, exist_ok=True)
        dump(search, '_'.join([
            args.results_dir, '/search', dataset_tr_name,
            args.slr_meth.lower(), args.fs_meth.lower(), args.clf_meth.lower()
        ]) + '.pkl')
    feature_idxs = np.arange(X_tr.shape[1])
    for step in search.best_estimator_.named_steps:
        if hasattr(search.best_estimator_.named_steps[step], 'get_support'):
            feature_idxs = feature_idxs[search.best_estimator_.named_steps[step].get_support(indices=True)]
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)[feature_idxs]
    weights = np.array([], dtype=float)
    if hasattr(search.best_estimator_.named_steps['clf'], 'coef_'):
        weights = np.square(search.best_estimator_.named_steps['clf'].coef_[0])
    elif hasattr(search.best_estimator_.named_steps['clf'], 'feature_importances_'):
        weights = search.best_estimator_.named_steps['clf'].feature_importances_
    elif ('fs2' in search.best_estimator_.named_steps):
        if (hasattr(search.best_estimator_.named_steps['fs2'], 'estimator_') and
            hasattr(search.best_estimator_.named_steps['fs2'].estimator_, 'coef_')):
            weights = np.square(search.best_estimator_.named_steps['fs2'].estimator_.coef_[0])
        elif hasattr(search.best_estimator_.named_steps['fs2'], 'scores_'):
            weights = search.best_estimator_.named_steps['fs2'].scores_
        elif hasattr(search.best_estimator_.named_steps['fs2'], 'feature_importances_'):
            weights = search.best_estimator_.named_steps['fs2'].feature_importances_
    print('Train:', dataset_tr_name, ' ROC AUC (CV): %.4f  BCR (CV): %.4f' % (
        search.cv_results_['mean_test_roc_auc'][search.best_index_],
        search.cv_results_['mean_test_bcr'][search.best_index_]
    ))
    print('Best Params:', search.best_params_)
    if weights.size > 0:
        print('Feature Rankings:')
        if args.show_annots:
            feature_ranks = sorted(
                zip(feature_idxs, feature_names, weights, r_eset_feature_annots(
                    eset_tr,
                    annots=robjects.StrVector(args.show_annots),
                    features=(feature_idxs + 1)
                )),
                key=itemgetter(2), reverse=True
            )
            for _, feature_name, weight, annot in feature_ranks:
                print(feature_name, '\t', weight, '\t', annot)
        else:
            feature_ranks = sorted(
                zip(feature_idxs, feature_names, weights),
                key=itemgetter(2), reverse=True
            )
            for _, feature_name, weight in feature_ranks:
                print(feature_name, '\t', weight)
    else:
        print('Features:')
        if args.show_annots:
            feature_ranks = sorted(
                zip(feature_idxs, feature_names, r_eset_feature_annots(
                    eset_tr,
                    annots=robjects.StrVector(args.show_annots),
                    features=(feature_idxs + 1)
                )),
                key=itemgetter(1)
            )
            for _, feature_name, annot in feature_ranks:
                print(feature_name, '\t', annot)
        else:
            feature_ranks = sorted(
                zip(feature_idxs, feature_names),
                key=itemgetter(1)
            )
            for _, feature_name in feature_ranks: print(feature_name)
    # plot grid search parameters vs cv perf metrics
    num_figures = 0
    sns.set_palette(sns.color_palette('hls', len(scv_scoring)))
    for param, param_values in param_grid.items():
        if len(param_values) == 1: continue
        new_shape = (
            len(param_values),
            int(len(search.cv_results_['params']) / len(param_values))
        )
        param_values_cv = np.ma.getdata(search.cv_results_['param_%s' % param])
        param_values_cv_sorted_idxs = np.where(
            np.array(param_values).reshape(len(param_values), 1) == param_values_cv
        )[1]
        plt.figure('Figure ' + str(args.analysis) + '-' + str(num_figures + 1))
        plt.rcParams['font.size'] = 14
        if param in params_num_xticks:
            x_axis = param_grid[param]
            plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
            plt.xticks(x_axis)
        elif param in params_fixed_xticks:
            x_axis = range(len(param_grid[param]))
            plt.xticks(x_axis, param_grid[param])
        plt.title(
            dataset_tr_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
            'Effect of ' + param + ' on CV Performance Metrics'
        )
        plt.xlabel(param)
        plt.ylabel('CV Score')
        for metric_idx, metric in enumerate(sorted(scv_scoring.keys(), reverse=True)):
            if args.scv_h_plt_meth == 'best':
                mean_scores_cv = np.transpose(np.reshape(
                    search.cv_results_[
                        'mean_test_%s' % metric
                    ][param_values_cv_sorted_idxs],
                    new_shape
                ))
                std_scores_cv = np.transpose(np.reshape(
                    search.cv_results_[
                        'std_test_%s' % metric
                    ][param_values_cv_sorted_idxs],
                    new_shape
                ))
                mean_scores_cv_max_idxs = np.argmax(mean_scores_cv, axis=0)
                mean_scores_cv = mean_scores_cv[
                    mean_scores_cv_max_idxs, np.arange(mean_scores_cv.shape[1])
                ]
                std_scores_cv = std_scores_cv[
                    mean_scores_cv_max_idxs, np.arange(std_scores_cv.shape[1])
                ]
            elif args.scv_h_plt_meth == 'all':
                all_scores_cv = np.array([])
                for split_idx in range(search.n_splits_):
                    split_scores_cv = np.transpose(np.reshape(
                        search.cv_results_[
                            'split%d_test_%s' % (split_idx, metric)
                        ][param_values_cv_sorted_idxs],
                        new_shape
                    ))
                    if all_scores_cv.size > 0:
                        all_scores_cv = np.vstack((all_scores_cv, split_scores_cv))
                    else:
                        all_scores_cv = split_scores_cv
                mean_scores_cv = np.mean(all_scores_cv, axis=0)
                std_scores_cv = np.std(all_scores_cv, axis=0)
            if metric_idx == 0:
                label = r'$\pm$ 1 std. dev.'
            else:
                label = None
            plt.plot(
                x_axis,
                mean_scores_cv,
                lw=2, alpha=0.8, label='Mean ' + metric.replace('_', ' ').upper()
            )
            plt.fill_between(
                x_axis,
                [m - s for m, s in zip(mean_scores_cv, std_scores_cv)],
                [m + s for m, s in zip(mean_scores_cv, std_scores_cv)],
                color='grey', alpha=0.2, label=label,
            )
            plt.legend(loc='lower right', fontsize='small')
            plt.grid('on')
            num_figures += 1
    # plot num top-ranked features selected vs test dataset perf metrics
    if args.dataset_te:
        dataset_te_basenames = natsorted(list(set(args.dataset_te) - set(args.dataset_tr)))
    else:
        dataset_te_basenames = natsorted(list(set(dataset_names) - set(args.dataset_tr)))
    if dataset_te_basenames:
        sns.set_palette(sns.color_palette('hls', len(dataset_te_basenames)))
        plt.figure('Figure ' + str(args.analysis) + '-' + str(num_figures + 1))
        plt.rcParams['font.size'] = 14
        plt.title(
            dataset_tr_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
            'Effect of Number of Top-Ranked Features Selected on Test Performance Metrics'
        )
        plt.xlabel('Number of top-ranked features selected')
        plt.ylabel('Test Score')
        x_axis = range(1, feature_idxs.size + 1)
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
        ranked_feature_idxs = [r[0] for r in feature_ranks]
        pipe = Pipeline(
            pipelines['slr'][args.slr_meth]['steps'] +
            pipelines['clf'][args.clf_meth]['steps']
        )
        pipe.set_params(
            **{ k: v for k, v in search.best_params_.items() if k.startswith('slr') or k.startswith('clf') }
        )
        if args.verbose > 1:
            print('Pipeline:')
            pprint(vars(pipe))
        for dataset_te_basename in dataset_te_basenames:
            if (len(args.dataset_tr) == 1 and not bc_meth) or args.no_addon_te:
                dataset_te_name = '_'.join([dataset_te_basename] + [x for x in prep_steps if x != 'mrg'])
            else:
                dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
            eset_te_name = 'eset_' + dataset_te_name
            eset_te_file = 'data/' + eset_te_name + '.Rda'
            if not path.isfile(eset_te_file): continue
            base.load(eset_te_file)
            eset_te = robjects.globalenv[eset_te_name]
            X_te = np.array(base.t(biobase.exprs(eset_te)), dtype=float)
            y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
            if args.filt_zero_feats:
                X_te = X_te[:, nzero_feature_idxs]
            if args.filt_zsdv_feats:
                X_te = X_te[:, nzsd_feature_idxs]
            if args.filt_nzvr_feats:
                X_te = X_te[:, nzvr_feature_idxs]
            if args.filt_ncor_feats:
                X_te = X_te[:, corr_feature_idxs]
            roc_aucs_te, bcrs_te = [], []
            for num_features in range(1, len(ranked_feature_idxs) + 1):
                top_feature_idxs = ranked_feature_idxs[:num_features]
                top_feature_names = ranked_feature_idxs[:num_features]
                pipe.fit(X_tr[:,top_feature_idxs], y_tr)
                if hasattr(pipe, 'decision_function'):
                    y_score = pipe.decision_function(X_te[:,top_feature_idxs])
                else:
                    y_score = pipe.predict_proba(X_te[:,top_feature_idxs])[:,1]
                roc_auc_te = roc_auc_score(y_te, y_score)
                fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
                y_pred = pipe.predict(X_te[:,top_feature_idxs])
                bcr_te = bcr_score(y_te, y_pred)
                roc_aucs_te.append(roc_auc_te)
                bcrs_te.append(bcr_te)
            plt.plot(
                x_axis, roc_aucs_te, lw=2, alpha=0.8,
                # label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, BCR = %0.4f $\pm$ %0.2f)' % (
                label=r'%s (ROC AUC = %0.4f, BCR = %0.4f)' % (
                    dataset_te_name,
                    np.max(roc_aucs_te), np.max(bcrs_te),
                    # np.mean(roc_aucs_te), np.std(roc_aucs_te),
                    # np.mean(bcrs_te), np.std(bcrs_te),
                ),
            )
            # plt.plot(x_axis, bcrs_te, lw=2, alpha=0.8)
        plt.legend(loc='lower right', fontsize='small')
        plt.grid('on')
        num_figures += 1
        # plot roc curve
        sns.set_palette(sns.color_palette('hls', len(dataset_te_basenames) + 1))
        plt.figure('Figure ' + str(args.analysis) + '-' + str(num_figures + 1))
        plt.rcParams['font.size'] = 14
        plt.title(
            dataset_tr_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
            'ROC Curve'
        )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        for dataset_te_basename in dataset_te_basenames:
            if (len(args.dataset_tr) == 1 and not bc_meth) or args.no_addon_te:
                dataset_te_name = '_'.join([dataset_te_basename] + [x for x in prep_steps if x != 'mrg'])
            else:
                dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
            eset_te_name = 'eset_' + dataset_te_name
            if not base.exists(eset_te_name)[0]: continue
            eset_te = robjects.globalenv[eset_te_name]
            X_te = np.array(base.t(biobase.exprs(eset_te)), dtype=float)
            y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
            if args.filt_zero_feats:
                X_te = X_te[:, nzero_feature_idxs]
            if args.filt_zsdv_feats:
                X_te = X_te[:, nzsd_feature_idxs]
            if args.filt_nzvr_feats:
                X_te = X_te[:, nzvr_feature_idxs]
            if args.filt_ncor_feats:
                X_te = X_te[:, corr_feature_idxs]
            if hasattr(search, 'decision_function'):
                y_score = search.decision_function(X_te)
            else:
                y_score = search.predict_proba(X_te)[:,1]
            roc_auc_te = roc_auc_score(y_te, y_score)
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            y_pred = search.predict(X_te)
            bcr_te = bcr_score(y_te, y_pred)
            plt.plot(
                fpr, tpr, lw=3, alpha=0.8,
                label=r'%s ROC (AUC = %0.4f, BCR = %0.4f)' % (dataset_te_name, roc_auc_te, bcr_te),
            )
            # print summary info
            print(
                'Test: %3s' % dataset_te_name,
                ' ROC AUC: %.4f' % roc_auc_te,
                ' BCR: %.4f' % bcr_te,
            )
        plt.plot([0, 1], [0, 1], linestyle='--', lw=3, alpha=0.2, label='Chance')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid('off')
elif args.analysis == 3:
    if args.data_type:
        data_types = [x for x in data_types if x in args.data_type]
    if args.norm_meth:
        norm_methods = [x for x in norm_methods if x in args.norm_meth]
    if args.feat_type:
        feat_types = [x for x in feat_types if x in args.feat_type]
    if args.prep_meth:
        prep_methods = [x for x in prep_methods if x in args.prep_meth]
    if args.bc_meth:
        bc_methods = [x for x in bc_methods if x in args.bc_meth]
    if args.filt_type:
        filt_types = [x for x in filt_types if x in args.filt_type]
    prep_groups, prep_group_info = [], []
    for data_type in data_types:
        for norm_meth in norm_methods:
            for feat_type in feat_types:
                for prep_meth in prep_methods:
                    for bc_meth in bc_methods:
                        for filt_type in filt_types:
                            prep_groups.append([
                                x for x in [
                                    data_type, norm_meth, feat_type, prep_meth, bc_meth, filt_type
                                ] if x not in ('None', 'none')
                            ])
                            prep_group_info.append({
                                'pkm': True if norm_meth == 'pkm' else False,
                                'mrg': True if prep_meth == 'mrg' else False,
                                'bcm': True if bc_meth not in ('None', 'none') else False,
                            })
    if args.fs_meth:
        pipelines['fs'] = { k: v for k, v in pipelines['fs'].items() if k in args.fs_meth }
    if args.slr_meth:
        pipelines['slr'] = { k: v for k, v in pipelines['slr'].items() if k in args.slr_meth }
    if args.clf_meth:
        pipelines['clf'] = { k: v for k, v in pipelines['clf'].items() if k in args.clf_meth }
    if args.scv_type == 'rand':
        args.fs_meth = args.fs_meth[0]
        args.slr_meth = args.slr_meth[0]
        args.clf_meth = args.clf_meth[0]
    if args.dataset_tr and args.num_combo_tr:
        dataset_tr_combos = [list(x) for x in combinations(natsorted(args.dataset_tr), args.num_combo_tr)]
    elif args.dataset_tr:
        dataset_tr_combos = [list(x) for x in combinations(natsorted(args.dataset_tr), len(args.dataset_tr))]
    else:
        dataset_tr_combos = [list(x) for x in combinations(natsorted(dataset_names), args.num_combo_tr)]
    if args.dataset_te:
        dataset_te_basenames = [x for x in natsorted(dataset_names) if x in args.dataset_te]
    else:
        dataset_te_basenames = dataset_names
    # determine which data combinations will be used
    num_dataset_pairs = 0
    dataset_tr_combos_subset, dataset_te_basenames_subset, prep_groups_subset = [], [], []
    for dataset_tr_combo in dataset_tr_combos:
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        for dataset_te_basename in dataset_te_basenames:
            if dataset_te_basename in dataset_tr_combo: continue
            for pr_idx, prep_steps in enumerate(prep_groups):
                prep_method = '_'.join(prep_steps)
                if len(dataset_tr_combo) > 1 or prep_group_info[pr_idx]['bcm']:
                    dataset_tr_name = '_'.join([dataset_tr_basename, prep_method, 'tr'])
                else:
                    dataset_tr_name = '_'.join([dataset_tr_basename, prep_method])
                if (len(dataset_tr_combo) == 1 and not prep_group_info[pr_idx]['bcm']) or args.no_addon_te:
                    dataset_te_name = '_'.join([dataset_te_basename] + [x for x in prep_steps if x != 'mrg'])
                else:
                    dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
                eset_tr_file = 'data/' + eset_tr_name + '.Rda'
                eset_te_file = 'data/' + eset_te_name + '.Rda'
                if not path.isfile(eset_tr_file) or not path.isfile(eset_te_file): continue
                if args.load_only: print(dataset_tr_name, '->', dataset_te_name)
                dataset_tr_combos_subset.append(dataset_tr_combo)
                dataset_te_basenames_subset.append(dataset_te_basename)
                prep_groups_subset.append(prep_steps)
                num_dataset_pairs += 1
    dataset_tr_combos = [x for x in dataset_tr_combos if x in dataset_tr_combos_subset]
    dataset_te_basenames = [x for x in dataset_te_basenames if x in dataset_te_basenames_subset]
    prep_groups = [x for x in prep_groups if x in prep_groups_subset]
    print('Num dataset pairs:', num_dataset_pairs)
    if args.load_only:
        if args.pipe_memory: rmtree(cachedir)
        quit()
    score_dtypes = [
        ('roc_auc_cv', float), ('bcr_cv', float),
        ('roc_auc_te', float), ('bcr_te', float),
        ('num_features', int),
    ]
    results = {
        'te_pr': np.zeros((len(dataset_te_basenames), len(prep_groups)), dtype=[
            ('tr_fs', [
                ('clf_slr', score_dtypes, (len(pipelines['clf']), len(pipelines['slr'])))
            ], (len(dataset_tr_combos), len(pipelines['fs'])))
        ]),
        'tr_pr': np.zeros((len(dataset_tr_combos), len(prep_groups)), dtype=[
            ('te_fs', [
                ('clf_slr', score_dtypes, (len(pipelines['clf']), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(pipelines['fs'])))
        ]),
        'te_fs': np.zeros((len(dataset_te_basenames), len(pipelines['fs'])), dtype=[
            ('tr_pr', [
                ('clf_slr', score_dtypes, (len(pipelines['clf']), len(pipelines['slr'])))
            ], (len(dataset_tr_combos), len(prep_groups)))
        ]),
        'tr_fs': np.zeros((len(dataset_tr_combos), len(pipelines['fs'])), dtype=[
            ('te_pr', [
                ('clf_slr', score_dtypes, (len(pipelines['clf']), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(prep_groups)))
        ]),
        'te_clf': np.zeros((len(dataset_te_basenames), len(pipelines['clf'])), dtype=[
            ('tr_pr', [
                ('fs_slr', score_dtypes, (len(pipelines['fs']), len(pipelines['slr'])))
            ], (len(dataset_tr_combos), len(prep_groups)))
        ]),
        'tr_clf': np.zeros((len(dataset_tr_combos), len(pipelines['clf'])), dtype=[
            ('te_pr', [
                ('fs_slr', score_dtypes, (len(pipelines['fs']), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(prep_groups)))
        ]),
        'pr_fs': np.zeros((len(prep_groups), len(pipelines['fs'])), dtype=[
            ('te_tr', [
                ('clf_slr', score_dtypes, (len(pipelines['clf']), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(dataset_tr_combos)))
        ]),
        'pr_clf': np.zeros((len(prep_groups), len(pipelines['clf'])), dtype=[
            ('te_tr', [
                ('fs_slr', score_dtypes, (len(pipelines['fs']), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(dataset_tr_combos)))
        ]),
        'fs_clf': np.zeros((len(pipelines['fs']), len(pipelines['clf'])), dtype=[
            ('te_tr', [
                ('pr_slr', score_dtypes, (len(prep_groups), len(pipelines['slr'])))
            ], (len(dataset_te_basenames), len(dataset_tr_combos)))
        ]),
    }
    dataset_pair_counter = 1
    for tr_idx, dataset_tr_combo in enumerate(dataset_tr_combos):
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        for te_idx, dataset_te_basename in enumerate(dataset_te_basenames):
            if dataset_te_basename in dataset_tr_combo: continue
            for pr_idx, prep_steps in enumerate(prep_groups):
                prep_method = '_'.join(prep_steps)
                if len(dataset_tr_combo) > 1 or prep_group_info[pr_idx]['bcm']:
                    dataset_tr_name = '_'.join([dataset_tr_basename, prep_method, 'tr'])
                else:
                    dataset_tr_name = '_'.join([dataset_tr_basename, prep_method])
                if (len(dataset_tr_combo) == 1 and not prep_group_info[pr_idx]['bcm']) or args.no_addon_te:
                    dataset_te_name = '_'.join([dataset_te_basename] + [x for x in prep_steps if x != 'mrg'])
                else:
                    dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
                eset_tr_file = 'data/' + eset_tr_name + '.Rda'
                eset_te_file = 'data/' + eset_te_name + '.Rda'
                if not path.isfile(eset_tr_file) or not path.isfile(eset_te_file): continue
                print(str(dataset_pair_counter), ': ', dataset_tr_name, ' -> ', dataset_te_name, sep='')
                base.load('data/' + eset_tr_name + '.Rda')
                eset_tr = robjects.globalenv[eset_tr_name]
                X_tr = np.array(base.t(biobase.exprs(eset_tr)), dtype=float)
                y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
                base.load('data/' + eset_te_name + '.Rda')
                eset_te = robjects.globalenv[eset_te_name]
                X_te = np.array(base.t(biobase.exprs(eset_te)), dtype=float)
                y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
                if args.filt_zero_feats:
                    nzero_feature_idxs = np.array(r_data_nzero_col_idxs(X_tr), dtype=int)
                    X_tr = X_tr[:, nzero_feature_idxs]
                    X_te = X_te[:, nzero_feature_idxs]
                if args.filt_zero_samples:
                    nzero_sample_idxs = np.array(r_data_nzero_col_idxs(X_tr.T), dtype=int)
                    X_tr = X_tr[nzero_sample_idxs, :]
                    y_tr = y_tr[nzero_sample_idxs]
                if args.filt_zsdv_feats:
                    nzsd_feature_idxs = np.array(r_data_nzsd_col_idxs(X_tr), dtype=int)
                    X_tr = X_tr[:, nzsd_feature_idxs]
                    X_te = X_te[:, nzsd_feature_idxs]
                if args.filt_zsdv_samples:
                    nzsd_sample_idxs = np.array(r_data_nzsd_col_idxs(X_tr.T), dtype=int)
                    X_tr = X_tr[nzsd_sample_idxs, :]
                    y_tr = y_tr[nzsd_sample_idxs]
                if args.filt_nzvr_feats:
                    nzvr_feature_idxs = np.array(r_data_nzvr_col_idxs(
                        X_tr, freqCut=args.filt_nzvr_feat_freq_cut, uniqueCut=args.filt_nzvr_feat_uniq_cut
                    ), dtype=int)
                    X_tr = X_tr[:, nzvr_feature_idxs]
                    X_te = X_te[:, nzvr_feature_idxs]
                if args.filt_nzvr_samples:
                    nzvr_sample_idxs = np.array(r_data_nzvr_col_idxs(
                        X_tr.T, freqCut=args.filt_nzvr_sample_freq_cut, uniqueCut=args.filt_nzvr_sample_uniq_cut
                    ), dtype=int)
                    X_tr = X_tr[nzvr_sample_idxs, :]
                    y_tr = y_tr[nzvr_sample_idxs]
                if args.filt_ncor_feats:
                    corr_feature_idxs = np.array(
                        r_data_corr_col_idxs(X_tr, cutoff=args.filt_ncor_feat_cut), dtype=int
                    )
                    X_tr = X_tr[:, corr_feature_idxs]
                    X_te = X_te[:, corr_feature_idxs]
                if args.filt_ncor_samples:
                    corr_sample_idxs = np.array(
                        r_data_corr_col_idxs(X_tr.T, cutoff=args.filt_ncor_sample_cut), dtype=int
                    )
                    X_tr = X_tr[corr_sample_idxs, :]
                    y_tr = y_tr[corr_sample_idxs]
                if args.scv_type == 'grid':
                    param_grid_idx = 0
                    param_grid, param_grid_data = [], []
                    for fs_idx, fs_meth in enumerate(pipelines['fs']):
                        fs_meth_pipeline = deepcopy(pipelines['fs'][fs_meth])
                        if fs_meth == 'Limma-KBest':
                            for (step, object) in fs_meth_pipeline['steps']:
                                if object.__class__.__name__ == 'SelectKBest':
                                    if prep_group_info[pr_idx]['pkm']:
                                        object.set_params(score_func=limma_pkm_score_func)
                                    else:
                                        object.set_params(score_func=limma_score_func)
                        for fs_params in fs_meth_pipeline['param_grid']:
                            for param in fs_params:
                                if param in params_feature_select:
                                    param_grid[param] = list(filter(
                                        lambda x: x <= min(
                                            X_tr.shape[1],
                                            args.fs_skb_k_max if args.fs_skb_lim_off else y_tr.shape[0]
                                        ),
                                        param_grid[param]
                                    ))
                            for slr_idx, slr_meth in enumerate(pipelines['slr']):
                                for slr_params in pipelines['slr'][slr_meth]['param_grid']:
                                    for clf_idx, clf_meth in enumerate(pipelines['clf']):
                                        for clf_params in pipelines['clf'][clf_meth]['param_grid']:
                                            params = { **fs_params, **slr_params, **clf_params }
                                            for (step, object) in \
                                                fs_meth_pipeline['steps'] + \
                                                pipelines['slr'][slr_meth]['steps'] + \
                                                pipelines['clf'][clf_meth]['steps'] \
                                            : params[step] = [ object ]
                                            param_grid.append(params)
                                            params_data = {
                                                'meth_idxs': {
                                                    'fs': fs_idx, 'slr': slr_idx, 'clf': clf_idx,
                                                },
                                                'grid_idxs': [],
                                            }
                                            for param_combo in ParameterGrid(params):
                                                params_data['grid_idxs'].append(param_grid_idx)
                                                param_grid_idx += 1
                                            param_grid_data.append(params_data)
                    search = GridSearchCV(
                        Pipeline(list(map(lambda x: (x, None), pipeline_order)), memory=memory),
                        param_grid=param_grid, scoring=scv_scoring, refit=args.scv_refit,
                        cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
                        error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
                    )
                elif args.scv_type == 'rand':
                    pipe = Pipeline(sorted(
                        pipelines['slr'][args.slr_meth]['steps'] +
                        pipelines['fs'][args.fs_meth]['steps'] +
                        pipelines['clf'][args.clf_meth]['steps'],
                        key=lambda s: pipeline_order.index(s[0])
                    ), memory=memory)
                    param_grid = {
                        **pipelines['slr'][args.slr_meth]['param_grid'][0],
                        **pipelines['fs'][args.fs_meth]['param_grid'][0],
                        **pipelines['clf'][args.clf_meth]['param_grid'][0],
                    }
                    if args.fs_meth == 'Limma-KBest':
                        if prep_group_info[pr_idx]['pkm']:
                            pipe.set_params(fs1__score_func=limma_pkm_score_func)
                        else:
                            pipe.set_params(fs1__score_func=limma_score_func)
                    for param in param_grid:
                        if param in params_feature_select:
                            param_grid[param] = list(filter(
                                lambda x: x <= min(
                                    X_tr.shape[1],
                                    args.fs_skb_k_max if args.fs_skb_lim_off else y_tr.shape[0]
                                ),
                                param_grid[param]
                            ))
                    search = RandomizedSearchCV(
                        pipe, param_distributions=param_grid, scoring=scv_scoring, refit=args.scv_refit,
                        cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
                        error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
                        n_iter=args.scv_n_iter,
                    )
                    if args.verbose > 0:
                        print('Pipeline:')
                        pprint(vars(pipe))
                if args.verbose > 0:
                    print('Param grid:')
                    pprint(param_grid)
                    if args.verbose > 1 and args.scv_type == 'grid':
                        print('Param grid data:')
                        pprint(param_grid_data)
                search.fit(X_tr, y_tr)
                group_best_grid_idx, group_best_params = [], []
                for group_idx, param_grid_group in enumerate(param_grid_data):
                    for grid_idx in param_grid_group['grid_idxs']:
                        if group_idx < len(group_best_grid_idx):
                            if (search.cv_results_['rank_test_' + args.scv_refit][grid_idx] <
                                search.cv_results_['rank_test_' + args.scv_refit][group_best_grid_idx[group_idx]]):
                                group_best_grid_idx[group_idx] = grid_idx
                        else:
                            group_best_grid_idx.append(grid_idx)
                    group_best_params.append({
                        k: clone(v) if k in pipeline_order else v
                        for k, v in search.cv_results_['params'][group_best_grid_idx[group_idx]].items()
                    })
                print('Fitting ' + str(len(group_best_params)) + ' pipelines', end='', flush=True)
                if args.scv_verbose > 0: print()
                pipes = Parallel(n_jobs=args.num_cores, verbose=args.scv_verbose)(
                    delayed(fit_pipeline)(params, pipeline_order, X_tr, y_tr) for params in group_best_params
                )
                if args.scv_verbose == 0: print('done')
                best_roc_auc_te = 0
                best_bcr_te = 0
                best_params_te = {}
                for group_idx, param_grid_group in enumerate(param_grid_data):
                    if hasattr(pipes[group_idx], 'decision_function'):
                        y_score = pipes[group_idx].decision_function(X_te)
                    else:
                        y_score = pipes[group_idx].predict_proba(X_te)[:,1]
                    roc_auc_te = roc_auc_score(y_te, y_score)
                    y_pred = pipes[group_idx].predict(X_te)
                    bcr_te = bcr_score(y_te, y_pred)
                    metric_scores_te = { 'roc_auc_te': roc_auc_te, 'bcr_te': bcr_te }
                    fs_idx = param_grid_group['meth_idxs']['fs']
                    clf_idx = param_grid_group['meth_idxs']['clf']
                    slr_idx = param_grid_group['meth_idxs']['slr']
                    for metric in scv_scoring.keys():
                        metric_cv = metric + '_cv'
                        metric_te = metric + '_te'
                        metric_score_cv = search.cv_results_['mean_test_' + metric][group_best_grid_idx[group_idx]]
                        (results['te_pr'][te_idx, pr_idx]['tr_fs'][tr_idx, fs_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['te_pr'][te_idx, pr_idx]['tr_fs'][tr_idx, fs_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['tr_pr'][tr_idx, pr_idx]['te_fs'][te_idx, fs_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['tr_pr'][tr_idx, pr_idx]['te_fs'][te_idx, fs_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['te_fs'][te_idx, fs_idx]['tr_pr'][tr_idx, pr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['te_fs'][te_idx, fs_idx]['tr_pr'][tr_idx, pr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['tr_fs'][tr_idx, fs_idx]['te_pr'][te_idx, pr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['tr_fs'][tr_idx, fs_idx]['te_pr'][te_idx, pr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['te_clf'][te_idx, clf_idx]['tr_pr'][tr_idx, pr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['te_clf'][te_idx, clf_idx]['tr_pr'][tr_idx, pr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['tr_clf'][tr_idx, clf_idx]['te_pr'][te_idx, pr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['tr_clf'][tr_idx, clf_idx]['te_pr'][te_idx, pr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['pr_fs'][pr_idx, fs_idx]['te_tr'][te_idx, tr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['pr_fs'][pr_idx, fs_idx]['te_tr'][te_idx, tr_idx]
                            ['clf_slr'][clf_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['pr_clf'][pr_idx, clf_idx]['te_tr'][te_idx, tr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['pr_clf'][pr_idx, clf_idx]['te_tr'][te_idx, tr_idx]
                            ['fs_slr'][fs_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                        (results['fs_clf'][fs_idx, clf_idx]['te_tr'][te_idx, tr_idx]
                            ['pr_slr'][pr_idx, slr_idx][metric_cv]) = metric_score_cv
                        (results['fs_clf'][fs_idx, clf_idx]['te_tr'][te_idx, tr_idx]
                            ['pr_slr'][pr_idx, slr_idx][metric_te]) = metric_scores_te[metric_te]
                    if ((args.scv_refit == 'roc_auc' and roc_auc_te > best_roc_auc_te) or
                        (args.scv_refit == 'bcr' and bcr_te > best_bcr_te)):
                        best_roc_auc_te = roc_auc_te
                        best_bcr_te = bcr_te
                        best_params_te = group_best_params[group_idx]
                best_grid_idx_cv = np.argmin(search.cv_results_['rank_test_' + args.scv_refit])
                best_roc_auc_cv = search.cv_results_['mean_test_roc_auc'][best_grid_idx_cv]
                best_bcr_cv = search.cv_results_['mean_test_bcr'][best_grid_idx_cv]
                best_params_cv = search.cv_results_['params'][best_grid_idx_cv]
                print('Best Params (Train):', best_params_cv)
                print('Best Params (Test):', best_params_te)
                print('ROC AUC (CV / Test): %.4f / %.4f' % (best_roc_auc_cv, best_roc_auc_te),
                    ' BCR (CV / Test): %.4f / %.4f' % (best_bcr_cv, best_bcr_te))
                base.remove(eset_tr_name)
                base.remove(eset_te_name)
                dataset_pair_counter += 1
                # flush cache with each tr/te pair run (can grow too big if not)
                if args.pipe_memory: memory.clear(warn=False)
    makedirs(args.results_dir, mode=0o755, exist_ok=True)
    dump(results, args.results_dir + '/results_analysis_' + str(args.analysis) + '.pkl')
    title_sub = ''
    if args.clf_meth and isinstance(args.clf_meth, str):
        title_sub = 'Classifier: ' + args.clf_meth
    if args.fs_meth and isinstance(args.fs_meth, str):
        if title_sub: title_sub += ' '
        title_sub = 'Feature Selection: ' + args.fs_meth
    if title_sub: title_sub = '[' + title_sub + ']'
    prep_methods = ['_'.join(g) for g in prep_groups]
    dataset_tr_basenames = ['_'.join(c) for c in dataset_tr_combos]
    figures = [
        # plot results['te_pr']
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_ticks_rotation': 15,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': results['te_pr'],
            'line_names': dataset_te_basenames,
            'field_results_key': 'tr_fs',
            'sub_results_key': 'clf_slr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': results['te_pr'].T,
            'line_names': prep_methods,
            'field_results_key': 'tr_fs',
            'sub_results_key': 'clf_slr',
        },
        # plot results['tr_pr']
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_ticks_rotation': 15,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': results['tr_pr'],
            'line_names': dataset_tr_basenames,
            'field_results_key': 'te_fs',
            'sub_results_key': 'clf_slr',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': results['tr_pr'].T,
            'line_names': prep_methods,
            'field_results_key': 'te_fs',
            'sub_results_key': 'clf_slr',
        },
        # plot results['te_fs']
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': results['te_fs'],
            'line_names': dataset_te_basenames,
            'field_results_key': 'tr_pr',
            'sub_results_key': 'clf_slr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': results['te_fs'].T,
            'line_names': list(pipelines['fs'].keys()),
            'field_results_key': 'tr_pr',
            'sub_results_key': 'clf_slr',
        },
        # plot results['tr_fs']
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': results['tr_fs'],
            'line_names': dataset_tr_basenames,
            'field_results_key': 'te_pr',
            'sub_results_key': 'clf_slr',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': results['tr_fs'].T,
            'line_names': list(pipelines['fs'].keys()),
            'field_results_key': 'te_pr',
            'sub_results_key': 'clf_slr',
        },
        # plot results['te_clf']
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': results['te_clf'],
            'line_names': dataset_te_basenames,
            'field_results_key': 'tr_pr',
            'sub_results_key': 'fs_slr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': results['te_clf'].T,
            'line_names': list(pipelines['clf'].keys()),
            'field_results_key': 'tr_pr',
            'sub_results_key': 'fs_slr',
        },
        # plot results['tr_clf']
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': results['tr_clf'],
            'line_names': dataset_tr_basenames,
            'field_results_key': 'te_pr',
            'sub_results_key': 'fs_slr',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': results['tr_clf'].T,
            'line_names': list(pipelines['clf'].keys()),
            'field_results_key': 'te_pr',
            'sub_results_key': 'fs_slr',
        },
        # plot results['pr_fs']
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': results['pr_fs'],
            'line_names': prep_methods,
            'field_results_key': 'te_tr',
            'sub_results_key': 'clf_slr',
        },
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_ticks_rotation': 15,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': results['pr_fs'].T,
            'line_names': list(pipelines['fs'].keys()),
            'field_results_key': 'te_tr',
            'sub_results_key': 'clf_slr',
        },
        # plot results['pr_clf']
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': results['pr_clf'],
            'line_names': prep_methods,
            'field_results_key': 'te_tr',
            'sub_results_key': 'fs_slr',
        },
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_ticks_rotation': 15,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': results['pr_clf'].T,
            'line_names': list(pipelines['clf'].keys()),
            'field_results_key': 'te_tr',
            'sub_results_key': 'fs_slr',
        },
        # plot results['fs_clf']
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': results['fs_clf'],
            'line_names': list(pipelines['fs'].keys()),
            'field_results_key': 'te_tr',
            'sub_results_key': 'pr_slr',
        },
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': results['fs_clf'].T,
            'line_names': list(pipelines['clf'].keys()),
            'field_results_key': 'te_tr',
            'sub_results_key': 'pr_slr',
        },
    ]
    plt.rcParams['figure.max_open_warning'] = 0
    for figure_idx, figure in enumerate(figures):
        if (args.fs_meth and len(args.fs_meth) == 1 and
            args.slr_meth and len(args.slr_meth) == 1 and
            args.clf_meth and len(args.clf_meth) == 1 and
            figure_idx > 3): continue
        legend_kwargs = {
            'loc': 'lower left',
            'ncol': max(1, len(figure['line_names']) // 12),
        }
        if len(figure['line_names']) > 10:
            legend_kwargs['fontsize'] = 'xx-small'
        else:
            legend_kwargs['fontsize'] = 'x-small'
        sns.set_palette(sns.color_palette('hls', len(figure['line_names'])))
        for metric_idx, metric in enumerate(sorted(scv_scoring.keys(), reverse=True)):
            metric_title = metric.replace('_', ' ').upper()
            figure_name = 'Figure ' + str(args.analysis) + '-' + str(figure_idx + 1) + '-' + str(metric_idx + 1)
            plt.figure(figure_name + 'A')
            plt.rcParams['font.size'] = 14
            plt.title(
                'Effect of ' + figure['x_axis_title'] + ' on Train CV ' +
                metric_title + ' for each ' + figure['lines_title'] + '\n' +
                figure['title_sub']
            )
            plt.xlabel(figure['x_axis_title'])
            plt.ylabel(metric_title)
            if 'x_ticks_rotation' in figure and len(figure['x_axis']) > 7:
                plt.xticks(
                    figure['x_axis'], figure['x_axis_labels'],
                    fontsize='x-small', rotation=figure['x_ticks_rotation'],
                )
            else:
                plt.xticks(figure['x_axis'], figure['x_axis_labels'], fontsize='small')
            if len(figure['x_axis']) > 20:
                plt.xlim([ min(figure['x_axis']) - 1, max(figure['x_axis']) + 1 ])
            plt.figure(figure_name + 'B')
            plt.rcParams['font.size'] = 14
            plt.title(
                'Effect of ' + figure['x_axis_title'] + ' on Test ' +
                metric_title + ' for each ' + figure['lines_title'] + '\n' +
                figure['title_sub']
            )
            plt.xlabel(figure['x_axis_title'])
            plt.ylabel(metric_title)
            if 'x_ticks_rotation' in figure and len(figure['x_axis']) > 7:
                plt.xticks(
                    figure['x_axis'], figure['x_axis_labels'],
                    fontsize='x-small', rotation=figure['x_ticks_rotation'],
                )
            else:
                plt.xticks(figure['x_axis'], figure['x_axis_labels'], fontsize='small')
            if len(figure['x_axis']) > 20:
                plt.xlim([ min(figure['x_axis']) - 1, max(figure['x_axis']) + 1 ])
            for row_idx, row_results in enumerate(figure['results']):
                mean_scores_cv = np.full((figure['results'].shape[1],), np.nan, dtype=float)
                range_scores_cv = np.full((2, figure['results'].shape[1]), np.nan, dtype=float)
                mean_scores_te = np.full((figure['results'].shape[1],), np.nan, dtype=float)
                range_scores_te = np.full((2, figure['results'].shape[1]), np.nan, dtype=float)
                num_features = np.array([], dtype=int)
                for col_idx, col_results in enumerate(row_results):
                    scores_cv = np.array([], dtype=float)
                    scores_te = np.array([], dtype=float)
                    field_results = col_results[figure['field_results_key']]
                    if 'sub_results_key' in figure:
                        sub_field_results = field_results[figure['sub_results_key']]
                        scores_cv = sub_field_results[metric + '_cv'][sub_field_results[metric + '_cv'] > 0]
                        scores_te = sub_field_results[metric + '_te'][sub_field_results[metric + '_te'] > 0]
                        num_features = np.append(num_features, sub_field_results['num_features'])
                    else:
                        scores_cv = field_results[metric + '_cv'][field_results[metric + '_cv'] > 0]
                        scores_te = field_results[metric + '_te'][field_results[metric + '_te'] > 0]
                        num_features = np.append(num_features, field_results['num_features'])
                    if scores_cv.size > 0:
                        mean_scores_cv[col_idx] = np.mean(scores_cv)
                        range_scores_cv[0][col_idx] = np.mean(scores_cv) - np.min(scores_cv)
                        range_scores_cv[1][col_idx] = np.max(scores_cv) - np.mean(scores_cv)
                        mean_scores_te[col_idx] = np.mean(scores_te)
                        range_scores_te[0][col_idx] = np.mean(scores_te) - np.min(scores_te)
                        range_scores_te[1][col_idx] = np.max(scores_te) - np.mean(scores_te)
                if not np.all(np.isnan(mean_scores_cv)):
                    label_values_cv = (
                        figure['line_names'][row_idx], 'CV',
                        np.mean(mean_scores_cv[~np.isnan(mean_scores_cv)]),
                        np.std(mean_scores_cv[~np.isnan(mean_scores_cv)]),
                    )
                    label_values_te = (
                        figure['line_names'][row_idx], 'Test',
                        np.mean(mean_scores_te[~np.isnan(mean_scores_te)]),
                        np.std(mean_scores_te[~np.isnan(mean_scores_te)]),
                    )
                    if np.mean(num_features) == 0:
                        label = r'%s (%s = %0.4f $\pm$ %0.2f)'
                    elif np.std(num_features) == 0:
                        label = r'%s (%s = %0.4f $\pm$ %0.2f, Features = %d)'
                        label_values_cv = label_values_cv + (np.mean(num_features),)
                        label_values_te = label_values_te + (np.mean(num_features),)
                    else:
                        label = r'%s (%s = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)'
                        label_values_cv = label_values_cv + (np.mean(num_features), np.std(num_features))
                        label_values_te = label_values_te + (np.mean(num_features), np.std(num_features))
                    # color = next(plt.gca()._get_lines.prop_cycler)['color']
                    plt.figure(figure_name + 'A')
                    plt.errorbar(
                        figure['x_axis'], mean_scores_cv, yerr=range_scores_cv, lw=2, alpha=0.8,
                        capsize=10, elinewidth=2, markeredgewidth=2, marker='s',
                        label=label % label_values_cv,
                    )
                    plt.figure(figure_name + 'B')
                    plt.errorbar(
                        figure['x_axis'], mean_scores_te, yerr=range_scores_te, lw=2, alpha=0.8,
                        capsize=10, elinewidth=2, markeredgewidth=2, marker='s',
                        label=label % label_values_te,
                    )
            plt.figure(figure_name + 'A')
            plt.legend(**legend_kwargs)
            plt.grid('on')
            plt.figure(figure_name + 'B')
            plt.legend(**legend_kwargs)
            plt.grid('on')
            if args.save_figs:
                dump(plt.figure(figure_name + 'A'),
                    args.results_dir + '/' + (figure_name + 'A').replace(' ', '_').lower() + '.pkl')
                dump(plt.figure(figure_name + 'B'),
                    args.results_dir + '/' + (figure_name + 'B').replace(' ', '_').lower() + '.pkl')
if args.show_figs or not args.save_figs: plt.show()
if args.pipe_memory: rmtree(cachedir)
