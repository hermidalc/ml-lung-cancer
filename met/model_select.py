#!/usr/bin/env python

import warnings
from argparse import ArgumentParser
from copy import deepcopy
from os import path
from pprint import pprint
from tempfile import mkdtemp
from shutil import rmtree
from natsort import natsorted
from itertools import combinations
import numpy as np
import rpy2.rinterface as rinterface
rinterface.set_initoptions((b'rpy2', b'--quiet', b'--no-save', b'--max-ppsize=500000'))
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
# from rpy2.robjects import pandas2ri
# import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, SelectFpr, SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import dump, Memory
from feature_selection import CFS, FCBF, ReliefF
import matplotlib.pyplot as plt
from matplotlib import style

# ignore QDA collinearity warnings
warnings.filterwarnings('ignore', category=UserWarning, message="^Variables are collinear")

# config
parser = ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--splits', type=int, default=10, help='num outer splits')
parser.add_argument('--test-size', type=float, default=0.3, help='outer splits test size')
parser.add_argument('--datasets-tr', type=str, nargs='+', help='datasets tr')
parser.add_argument('--datasets-te', type=str, nargs='+', help='datasets te')
parser.add_argument('--num-tr-combo', type=int, default=1, help='dataset tr num combos')
parser.add_argument('--corr-cutoff', type=float, help='correlation filter cutoff')
parser.add_argument('--norm-meth', type=str, nargs='+', help='normalization method')
parser.add_argument('--no-addon-te', default=False, action='store_true', help='dataset te no addon')
parser.add_argument('--fs-meth', type=str, nargs='+', help='feature selection method')
parser.add_argument('--slr-meth', type=str, nargs='+', help='scaling method')
parser.add_argument('--clf-meth', type=str, nargs='+', help='classifier method')
parser.add_argument('--slr-mms-fr-min', type=int, nargs='+', help='slr mms fr min')
parser.add_argument('--slr-mms-fr-max', type=int, nargs='+', help='slr mms fr max')
parser.add_argument('--fs-skb-k', type=int, nargs='+', help='fs skb k select')
parser.add_argument('--fs-skb-k-max', type=int, default=1000, help='fs skb k select max')
parser.add_argument('--fs-sfp-p', type=float, nargs='+', help='fs sfp fpr')
parser.add_argument('--fs-sfm-ext-thres', type=float, nargs='+', help='fs sfm ext threshold')
parser.add_argument('--fs-sfm-ext-e', type=int, nargs='+', help='fs sfm ext n estimators')
parser.add_argument('--fs-sfm-ext-e-max', type=int, default=50, help='fs sfm ext n estimators max')
parser.add_argument('--fs-sfm-ext-d', type=int, nargs='+', help='fs sfm ext max depth')
parser.add_argument('--fs-sfm-ext-d-max', type=int, default=10, help='fs sfm ext max depth max')
parser.add_argument('--fs-sfm-ext-cw', type=str, nargs='+', help='fs sfm ext class weight')
parser.add_argument('--fs-sfm-svm-thres', type=float, nargs='+', help='fs sfm svm threshold')
parser.add_argument('--fs-sfm-svm-c', type=float, nargs='+', help='fs sfm svm c')
parser.add_argument('--fs-sfm-svm-cw', type=str, nargs='+', help='fs sfm svm class weight')
parser.add_argument('--fs-rfe-svm-c', type=float, nargs='+', help='fs rfe svm c')
parser.add_argument('--fs-rfe-svm-cw', type=str, nargs='+', help='fs rfe svm class weight')
parser.add_argument('--fs-rfe-ext-e', type=int, nargs='+', help='fs rfe ext n estimators')
parser.add_argument('--fs-rfe-ext-e-max', type=int, default=50, help='fs rfe ext n estimators max')
parser.add_argument('--fs-rfe-ext-d', type=int, nargs='+', help='fs rfe ext max depth')
parser.add_argument('--fs-rfe-ext-d-max', type=int, default=10, help='fs rfe ext max depth max')
parser.add_argument('--fs-rfe-ext-cw', type=str, nargs='+', help='fs rfe ext class weight')
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
parser.add_argument('--clf-svm-cache', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--clf-knn-k', type=int, nargs='+', help='clf knn neighbors')
parser.add_argument('--clf-knn-k-max', type=int, default=10, help='clf knn neighbors max')
parser.add_argument('--clf-knn-w', type=str, nargs='+', help='clf knn weights')
parser.add_argument('--clf-ext-e', type=int, nargs='+', help='clf ext n estimators')
parser.add_argument('--clf-ext-e-max', type=int, default=50, help='clf ext n estimators max')
parser.add_argument('--clf-ext-d', type=int, nargs='+', help='clf ext max depth')
parser.add_argument('--clf-ext-d-max', type=int, default=10, help='clf ext max depth max')
parser.add_argument('--clf-ext-cw', type=str, nargs='+', help='clf ext class weight')
parser.add_argument('--clf-ada-e', type=int, nargs='+', help='clf ada n estimators')
parser.add_argument('--clf-ada-e-max', type=int, default=200, help='clf ada n estimators max')
parser.add_argument('--clf-ada-lgr-c', type=float, nargs='+', help='clf ada lgr c')
parser.add_argument('--clf-ada-lgr-cw', type=str, nargs='+', help='clf ada lgr class weight')
parser.add_argument('--clf-grb-e', type=int, nargs='+', help='clf grb n estimators')
parser.add_argument('--clf-grb-e-max', type=int, default=300, help='clf grb n estimators max')
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
parser.add_argument('--num-cores', type=int, default=-1, help='num parallel cores')
parser.add_argument('--pipe-memory', default=False, action='store_true', help='turn on pipeline memory')
parser.add_argument('--save-model', default=False, action='store_true', help='save model')
parser.add_argument('--cache-dir', type=str, default='/tmp', help='cache dir')
parser.add_argument('--verbose', type=int, default=1, help='program verbosity')
args = parser.parse_args()
if args.test_size > 1.0: args.test_size = int(args.test_size)
if args.scv_size > 1.0: args.scv_size = int(args.scv_size)

base = importr('base')
base.source('functions.R')
# r_dataset_x = robjects.globalenv['datasetX']
r_dataset_y = robjects.globalenv['datasetY']
# r_dataset_nstd_idxs = robjects.globalenv['datasetNonZeroStdIdxs']
# r_dataset_corr_idxs = robjects.globalenv['datasetCorrIdxs']
# r_limma_feature_score = robjects.globalenv['limmaFeatureScore']
# r_limma_fpkm_feature_score = robjects.globalenv['limmaFpkmFeatureScore']
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

class CachedLogisticRegression(CachedFitMixin, LogisticRegression):
    pass

# def filter_constant_features(X):
#     keep_cols = []
#     for col in range(X.shape[1]):
#         _, count = np.unique(X[:,col], return_counts=True)
#         if count.size > 1: keep_cols.append(col)
#     # print(X[:,keep_cols].shape)
#     return X[:,keep_cols]

# limma feature selection scoring function
def limma(X, y):
    f, pv = r_limma_feature_score(X, y)
    return np.array(f), np.array(pv)

def limma_fpkm(X, y):
    f, pv = r_limma_fpkm_feature_score(X, y)
    return np.array(f), np.array(pv)

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

# config
if args.pipe_memory:
    limma_score_func = memory.cache(limma)
    limma_fpkm_score_func = memory.cache(limma_fpkm)
    f_classif_func = memory.cache(f_classif)
    mi_classif_func = memory.cache(mutual_info_classif)
    rfe_svm_estimator = CachedLinearSVC()
    sfm_svm_estimator = CachedLinearSVC(penalty='l1', dual=False)
    sfm_ext_estimator = CachedExtraTreesClassifier()
else:
    limma_score_func = limma
    limma_fpkm_score_func = limma_fpkm
    f_classif_func = f_classif
    mi_classif_func = mutual_info_classif
    rfe_svm_estimator = LinearSVC()
    sfm_svm_estimator = LinearSVC(penalty='l1', dual=False)
    sfm_ext_estimator = ExtraTreesClassifier()

scv_scoring = { 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }

# specify elements in sort order (needed by code dealing with gridsearch cv_results)
if args.slr_mms_fr_min and args.slr_mms_fr_max:
    SLR_MMS_FR = list(zip(args.slr_mms_fr_min, args.slr_mms_fr_max))
else:
    SLR_MMS_FR = [(0,1)]
if args.fs_skb_k:
    FS_SKB_K = sorted(args.fs_skb_k)
else:
    FS_SKB_K = list(range(1, args.fs_skb_k_max + 1, 1))
if args.fs_sfp_p:
    FS_SFP_P = sorted(args.fs_sfp_p)
else:
    FS_SFP_P = [ 1e-2, 5e-2 ]
if args.fs_sfm_ext_thres:
    FS_SFM_EXT_THRES = sorted(args.fs_sfm_ext_thres)
else:
    FS_SFM_EXT_THRES = np.logspace(-11, -5, 7)
if args.fs_sfm_ext_e:
    FS_SFM_EXT_E = sorted(args.fs_sfm_ext_e)
else:
    FS_SFM_EXT_E = list(range(5, args.fs_sfm_ext_e_max + 1, 5))
if args.fs_sfm_ext_d:
    FS_SFM_EXT_D = sorted(args.fs_sfm_ext_d)
else:
    FS_SFM_EXT_D = list(range(1, args.fs_sfm_ext_d_max + 1, 1)) + [None]
if args.fs_sfm_ext_cw:
    FS_SFM_EXT_CW = [None if a in ('None', 'none') else a for a in sorted(args.fs_sfm_ext_cw)]
else:
    FS_SFM_EXT_CW = ['balanced', None]
if args.fs_sfm_svm_thres:
    FS_SFM_SVM_THRES = sorted(args.fs_sfm_svm_thres)
else:
    FS_SFM_SVM_THRES = np.logspace(-11, -5, 7)
if args.fs_sfm_svm_c:
    FS_SFM_SVM_C = sorted(args.fs_sfm_svm_c)
else:
    FS_SFM_SVM_C = np.logspace(3, 7, 5)
if args.fs_sfm_svm_cw:
    FS_SFM_SVM_CW = [None if a in ('None', 'none') else a for a in sorted(args.fs_sfm_svm_cw)]
else:
    FS_SFM_SVM_CW = ['balanced', None]
if args.fs_rfe_svm_c:
    FS_RFE_SVM_C = sorted(args.fs_rfe_svm_c)
else:
    FS_RFE_SVM_C = np.logspace(-7, 3, 11)
if args.fs_rfe_svm_cw:
    FS_RFE_SVM_CW = [None if a in ('None', 'none') else a for a in sorted(args.fs_rfe_svm_cw)]
else:
    FS_RFE_SVM_CW = ['balanced', None]
if args.fs_rfe_ext_e:
    FS_RFE_EXT_E = sorted(args.fs_rfe_ext_e)
else:
    FS_RFE_EXT_E = list(range(5, args.fs_rfe_ext_e_max + 1, 5))
if args.fs_rfe_ext_d:
    FS_RFE_EXT_D = sorted(args.fs_rfe_ext_d)
else:
    FS_RFE_EXT_D = list(range(1, args.fs_rfe_ext_d_max + 1, 1)) + [None]
if args.fs_rfe_ext_cw:
    FS_RFE_EXT_CW = [None if a in ('None', 'none') else a for a in sorted(args.fs_rfe_ext_cw)]
else:
    FS_RFE_EXT_CW = ['balanced', None]
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
    CLF_SVM_CW = [None if a in ('None', 'none') else a for a in sorted(args.clf_svm_cw)]
else:
    CLF_SVM_CW = ['balanced', None]
if args.clf_svm_kern:
    CLF_SVM_KERN = sorted(args.clf_svm_kern)
else:
    CLF_SVM_KERN = ['linear', 'poly', 'rbf', 'sigmoid']
if args.clf_knn_k:
    CLF_KNN_K = sorted(args.clf_knn_k)
else:
    CLF_KNN_K = list(range(1, args.clf_knn_k_max + 1, 1))
if args.clf_knn_w:
    CLF_KNN_W = sorted(args.clf_knn_w)
else:
    CLF_KNN_W = ['distance', 'uniform']
if args.clf_ext_e:
    CLF_EXT_E = sorted(args.clf_ext_e)
else:
    CLF_EXT_E = list(range(5, args.clf_ext_e_max + 1, 5))
if args.clf_ext_d:
    CLF_EXT_D = sorted(args.clf_ext_d)
else:
    CLF_EXT_D = list(range(1, args.clf_ext_d_max + 1, 1)) + [None]
if args.clf_ext_cw:
    CLF_EXT_CW = [None if a in ('None', 'none') else a for a in sorted(args.clf_ext_cw)]
else:
    CLF_EXT_CW = ['balanced', None]
if args.clf_ada_e:
    CLF_ADA_E = sorted(args.clf_ada_e)
else:
    CLF_ADA_E = list(range(20, args.clf_ada_e_max + 1, 20))
if args.clf_ada_lgr_c:
    CLF_ADA_LGR_C = sorted(args.clf_ada_lgr_c)
else:
    CLF_ADA_LGR_C = np.logspace(-7, 3, 11)
if args.clf_ada_lgr_cw:
    CLF_ADA_LGR_CW = [None if a in ('None', 'none') else a for a in sorted(args.clf_ada_lgr_cw)]
else:
    CLF_ADA_LGR_CW = ['balanced', None]
if args.clf_grb_e:
    CLF_GRB_E = sorted(args.clf_grb_e)
else:
    CLF_GRB_E = list(range(5, args.clf_grb_e_max + 1, 5))
if args.clf_grb_d:
    CLF_GRB_D = sorted(args.clf_grb_d)
else:
    CLF_GRB_D = list(range(1, args.clf_grb_d_max + 1, 1))
if args.clf_grb_f:
    CLF_GRB_F = [None if a in ('None', 'none') else a for a in sorted(args.clf_grb_f)]
else:
    CLF_GRB_F = ['auto', 'sqrt', 'log2', None]
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
                ('fs2', SelectKBest(f_classif_func)),
            ],
            'param_grid': [
                {
                    'fs2__k': FS_SKB_K,
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
        'ExtraTrees-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(sfm_ext_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_SFM_EXT_E,
                    'fs2__estimator__max_depth': FS_SFM_EXT_D,
                    'fs2__estimator__class_weight': FS_SFM_EXT_CW,
                    'fs2__k': FS_SKB_K,
                },
            ],
        },
        'SVM-RFE': {
            'steps': [
                ('fs2', RFE(rfe_svm_estimator, verbose=args.fs_rfe_verbose)),
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
        'ExtraTrees-RFE': {
            'steps': [
                ('fs2', RFE(sfm_ext_estimator, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': FS_RFE_EXT_E,
                    'fs2__estimator__max_depth': FS_RFE_EXT_D,
                    'fs2__estimator__class_weight': FS_RFE_EXT_CW,
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
        'LinearSVC': {
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
        'SVC': {
            'steps': [
                ('clf', SVC(cache_size=args.clf_svm_cache)),
            ],
            'param_grid': [
                {
                    'clf__kernel': CLF_SVM_KERN,
                    'clf__C': CLF_SVM_C,
                    'clf__class_weight': CLF_SVM_CW,
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
                    'clf__max_depth': CLF_EXT_D,
                    'clf__class_weight': CLF_EXT_CW,
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
                    'clf__class_weight': CLF_EXT_CW,
                },
            ],
        },
        'RandomForest': {
            'steps': [
                ('clf', RandomForestClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_EXT_E,
                    'clf__max_depth': CLF_EXT_D,
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
        # 'QDA': {
        #     'steps': [
        #         ('clf', QuadraticDiscriminantAnalysis()),
        #     ],
        #     'param_grid': [
        #         { },
        #     ],
        # },
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

dataset_names = [
    'met_lung_dx',
]
norm_methods = [
    'none',
]
if args.analysis == 1:
    if args.norm_meth and args.norm_meth != 'none':
        norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
    if args.fs_meth:
        pipelines['fs'] = { k: v for k, v in pipelines['fs'].items() if k in args.fs_meth }
    if args.slr_meth:
        pipelines['slr'] = { k: v for k, v in pipelines['slr'].items() if k in args.slr_meth }
    if args.clf_meth:
        pipelines['clf'] = { k: v for k, v in pipelines['clf'].items() if k in args.clf_meth }
    if args.datasets_tr and args.num_tr_combo:
        dataset_tr_combos = [list(x) for x in combinations(
            [x for x in dataset_names if x in args.datasets_tr], args.num_tr_combo
        )]
    elif args.datasets_tr:
        dataset_tr_combos = [x for x in dataset_names if x in args.datasets_tr]
    else:
        dataset_tr_combos = [list(x) for x in combinations(dataset_names, args.num_tr_combo)]
    if args.datasets_te:
        dataset_te_names = [x for x in dataset_names if x in args.datasets_te]
    else:
        dataset_te_names = dataset_names
    for dataset_tr_combo in dataset_tr_combos:
        dataset_tr_nums = np.array(
            [i for i,d in enumerate(dataset_names) if d in dataset_tr_combo]
        ) + 1
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        if args.norm_meth and args.norm_meth != 'none':
            dataset_tr_name = '_'.join([dataset_tr_basename, norm_meth, 'tr'])
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            if len(dataset_tr_combo) > 1:
                df_p_tr_name = '_'.join(['df', 'p', dataset_tr_basename, 'tr'])
            else:
                df_p_tr_name = '_'.join(['df', 'p', dataset_tr_basename])
        elif len(dataset_tr_combo) > 1:
            dataset_tr_name = '_'.join([dataset_tr_basename, 'tr'])
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            df_p_tr_name = '_'.join(['df', 'p', dataset_tr_name])
        else:
            dataset_tr_name = dataset_tr_basename
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            df_p_tr_name = '_'.join(['df', 'p', dataset_tr_name])
        if (np.array(base.exists(df_X_tr_name), dtype=bool)[0] == False):
            base.load('data/' + df_X_tr_name + '.Rda')
        if (np.array(base.exists(df_p_tr_name), dtype=bool)[0] == False):
            base.load('data/' + df_p_tr_name + '.Rda')
        df_X_tr = robjects.globalenv[df_X_tr_name]
        df_p_tr = robjects.globalenv[df_p_tr_name]
        X_tr = np.array(base.as_matrix(df_X_tr))
        y_tr = np.array(r_dataset_y(df_p_tr), dtype=int)
        # nstd_feature_idxs = np.array(r_dataset_nstd_idxs(X_tr, samples=False), dtype=int)
        # nstd_sample_idxs = np.array(r_dataset_nstd_idxs(X_tr, samples=True), dtype=int)
        # X_tr = X_tr[np.ix_(nstd_sample_idxs, nstd_feature_idxs)]
        # y_tr = y_tr[nstd_sample_idxs]
        # if args.corr_cutoff:
        #     corr_feature_idxs = np.array(r_dataset_corr_idxs(X_tr, cutoff=args.corr_cutoff, samples=False), dtype=int)
        #     corr_sample_idxs = np.array(r_dataset_corr_idxs(X_tr, cutoff=args.corr_cutoff, samples=True), dtype=int)
        #     X_tr = X_tr[np.ix_(corr_sample_idxs, corr_feature_idxs)]
        #     y_tr = y_tr[corr_sample_idxs]
        print('Train:', dataset_tr_name, X_tr.shape, y_tr.shape)
        if args.scv_type == 'grid':
            param_grid_idx = 0
            param_grid, param_grid_data = [], []
            for fs_idx, fs_meth in enumerate(pipelines['fs']):
                fs_meth_pipeline = deepcopy(pipelines['fs'][fs_meth])
                if fs_meth == 'Limma-KBest':
                    for (step, object) in fs_meth_pipeline['steps']:
                        if object.__class__.__name__ == 'SelectKBest':
                            if dataset_tr_basename in rna_seq_fpkm_dataset_names:
                                object.set_params(score_func=limma_fpkm_score_func)
                            else:
                                object.set_params(score_func=limma_score_func)
                for fs_params in fs_meth_pipeline['param_grid']:
                    for param in fs_params:
                        if param in ('fs1__k', 'fs2__k', 'fs2__n_features_to_select'):
                            fs_params[param] = list(
                                filter(lambda x: x <= min(X_tr.shape[1], y_tr.shape[0]), fs_params[param])
                            )
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
            args.fs_meth = args.fs_meth[0]
            args.slr_meth = args.slr_meth[0]
            args.clf_meth = args.clf_meth[0]
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
                if dataset_tr_basename in rna_seq_fpkm_dataset_names:
                    pipe.set_params(fs1__score_func=limma_fpkm_score_func)
                else:
                    pipe.set_params(fs1__score_func=limma_score_func)
            for param in param_grid:
                if param in ('fs1__k', 'fs2__k', 'fs2__n_features_to_select'):
                    param_grid[param] = list(filter(lambda x: x <= min(X_tr.shape[1], y_tr.shape[0]), param_grid[param]))
            search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, scoring=scv_scoring, n_iter=args.scv_n_iter, refit=args.scv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
            )
            if args.verbose > 1:
                print('Pipeline:')
                pprint(vars(pipe))
        if args.verbose > 1:
            print('Param grid:')
            pprint(param_grid)
        search.fit(X_tr, y_tr)
        print('Train:', dataset_tr_name, ' ROC AUC (CV): %.4f  BCR (CV): %.4f' % (
            search.cv_results_['mean_test_roc_auc'][search.best_index_],
            search.cv_results_['mean_test_bcr'][search.best_index_]
        ))
        print('Best Params:', search.best_params_)
        feature_idxs = np.arange(X_tr.shape[1])
        for step in search.best_estimator_.named_steps:
            if hasattr(search.best_estimator_.named_steps[step], 'get_support'):
                feature_idxs = feature_idxs[search.best_estimator_.named_steps[step].get_support(indices=True)]
        feature_names = np.array(base.colnames(df_X_tr), dtype=str)[feature_idxs]
        weights = np.array([], dtype=float)
        if hasattr(search.best_estimator_.named_steps['clf'], 'coef_'):
            weights = np.square(search.best_estimator_.named_steps['clf'].coef_[0])
        elif hasattr(search.best_estimator_.named_steps['clf'], 'feature_importances_'):
            weights = search.best_estimator_.named_steps['clf'].feature_importances_
        elif (hasattr(search.best_estimator_.named_steps['fs2'], 'estimator_') and
            hasattr(search.best_estimator_.named_steps['fs2'].estimator_, 'coef_')):
            weights = np.square(search.best_estimator_.named_steps['fs2'].estimator_.coef_[0])
        elif hasattr(search.best_estimator_.named_steps['fs2'], 'scores_'):
            weights = search.best_estimator_.named_steps['fs2'].scores_
        elif hasattr(search.best_estimator_.named_steps['fs2'], 'feature_importances_'):
            weights = search.best_estimator_.named_steps['fs2'].feature_importances_
        if weights.size > 0:
            feature_ranks = sorted(zip(weights, feature_idxs, feature_names), reverse=True)
            print('Feature Rankings:')
            for weight, _, feature in feature_ranks: print(feature, '\t', weight)
        else:
            feature_ranks = sorted(zip(feature_idxs, feature_names), reverse=True)
            print('Features:')
            for _, feature in feature_ranks: print(feature)
        for dataset_te_basename in natsorted(list(set(dataset_te_names) - set(dataset_tr_combo))):
            dataset_te_num = np.array(
                [i for i,d in enumerate(dataset_names) if d == dataset_te_basename]
            ) + 1
            if args.norm_meth and args.norm_meth != 'none':
                dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
            else:
                dataset_te_name = dataset_te_basename
            df_X_te_name = '_'.join(['df', 'X', dataset_te_name])
            df_p_te_name = '_'.join(['df', 'p', dataset_te_basename])
            df_X_te_file = 'data/' + df_X_te_name + '.Rda'
            df_p_te_file = 'data/' + df_p_te_name + '.Rda'
            if not path.isfile(df_X_te_file): continue
            if (np.array(base.exists(df_X_te_name), dtype=bool)[0] == False):
                base.load(df_X_te_file)
            if (np.array(base.exists(df_p_te_name), dtype=bool)[0] == False):
                base.load(df_p_te_file)
            df_X_te = robjects.globalenv[df_X_te_name]
            df_p_te = robjects.globalenv[df_p_te_name]
            X_te = np.array(base.as_matrix(df_X_te))
            y_te = np.array(r_dataset_y(df_p_te), dtype=int)
            # X_te = X_te[:, nstd_feature_idxs]
            # if args.corr_cutoff:
            #     X_te = X_te[:, corr_feature_idxs]
            if hasattr(search, 'decision_function'):
                y_score = search.decision_function(X_te)
                roc_auc_te = roc_auc_score(y_te, y_score)
                # fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            else:
                probas = search.predict_proba(X_te)
                roc_auc_te = roc_auc_score(y_te, probas[:,1])
                # fpr, tpr, thres = roc_curve(y_te, probas[:,1], pos_label=1)
            y_pred = search.predict(X_te)
            bcr_te = bcr_score(y_te, y_pred)
            print('Test:', dataset_te_name, ' ROC AUC: %.6f  BCR: %.6f' % (roc_auc_te, bcr_te))
        if args.save_model:
            dump(search, 'results/search_' + dataset_tr_name.lower() + '.pkl')
        # flush cache with each combo run (grows too big if not)
        if args.pipe_memory: memory.clear(warn=False)
elif args.analysis == 2:
    if args.norm_meth and args.norm_meth != 'none':
        norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
    if args.fs_meth:
        pipelines['fs'] = { k: v for k, v in pipelines['fs'].items() if k in args.fs_meth }
    if args.slr_meth:
        pipelines['slr'] = { k: v for k, v in pipelines['slr'].items() if k in args.slr_meth }
    if args.clf_meth:
        pipelines['clf'] = { k: v for k, v in pipelines['clf'].items() if k in args.clf_meth }
    if args.datasets_tr and args.num_tr_combo:
        dataset_tr_combos = [list(x) for x in combinations(
            [x for x in dataset_names if x in args.datasets_tr], args.num_tr_combo
        )]
    elif args.datasets_tr:
        dataset_tr_combos = [x for x in dataset_names if x in args.datasets_tr]
    else:
        dataset_tr_combos = [list(x) for x in combinations(dataset_names, args.num_tr_combo)]
    if args.datasets_te:
        dataset_te_names = [x for x in dataset_names if x in args.datasets_te]
    else:
        dataset_te_names = dataset_names
    args.fs_meth = args.fs_meth[0]
    args.slr_meth = args.slr_meth[0]
    args.clf_meth = args.clf_meth[0]
    for dataset_tr_combo in dataset_tr_combos:
        dataset_tr_nums = np.array(
            [i for i,d in enumerate(dataset_names) if d in dataset_tr_combo]
        ) + 1
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        if args.norm_meth and args.norm_meth != 'none':
            dataset_tr_name = '_'.join([dataset_tr_basename, norm_meth, 'tr'])
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            if len(dataset_tr_combo) > 1:
                df_p_tr_name = '_'.join(['df', 'p', dataset_tr_basename, 'tr'])
            else:
                df_p_tr_name = '_'.join(['df', 'p', dataset_tr_basename])
        elif len(dataset_tr_combo) > 1:
            dataset_tr_name = '_'.join([dataset_tr_basename, 'tr'])
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            df_p_tr_name = '_'.join(['df', 'p', dataset_tr_name])
        else:
            dataset_tr_name = dataset_tr_basename
            df_X_tr_name = '_'.join(['df', 'X', dataset_tr_name])
            df_p_tr_name = '_'.join(['df', 'p', dataset_tr_name])
        if (np.array(base.exists(df_X_tr_name), dtype=bool)[0] == False):
            base.load('data/' + df_X_tr_name + '.Rda')
        if (np.array(base.exists(df_p_tr_name), dtype=bool)[0] == False):
            base.load('data/' + df_p_tr_name + '.Rda')
        df_X_tr = robjects.globalenv[df_X_tr_name]
        df_p_tr = robjects.globalenv[df_p_tr_name]
        X = np.array(base.as_matrix(df_X_tr))
        y = np.array(r_dataset_y(df_p_tr), dtype=int)
        # nstd_feature_idxs = np.array(r_dataset_nstd_idxs(X, samples=False), dtype=int)
        # nstd_sample_idxs = np.array(r_dataset_nstd_idxs(X, samples=True), dtype=int)
        # X = X[np.ix_(nstd_sample_idxs, nstd_feature_idxs)]
        # y = y[nstd_sample_idxs]
        # if args.corr_cutoff:
        #     corr_feature_idxs = np.array(r_dataset_corr_idxs(X, cutoff=args.corr_cutoff, samples=False), dtype=int)
        #     corr_sample_idxs = np.array(r_dataset_corr_idxs(X, cutoff=args.corr_cutoff, samples=True), dtype=int)
        #     X = X[np.ix_(corr_sample_idxs, corr_feature_idxs)]
        #     y = y[corr_sample_idxs]
        print('Dataset:', dataset_tr_name, X.shape, y.shape)
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
            if dataset_tr_basename in rna_seq_fpkm_dataset_names:
                pipe.set_params(fs1__score_func=limma_fpkm_score_func)
            else:
                pipe.set_params(fs1__score_func=limma_score_func)
        for param in param_grid:
            if param in ('fs1__k', 'fs2__k', 'fs2__n_features_to_select'):
                param_grid[param] = list(filter(lambda x: x <= min(X.shape[1], y.shape[0]), param_grid[param]))
        if args.scv_type == 'grid':
            search = GridSearchCV(
                pipe, param_grid=param_grid, scoring=scv_scoring, refit=args.scv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
            )
        elif args.scv_type == 'rand':
            search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, scoring=scv_scoring, n_iter=args.scv_n_iter, refit=args.scv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.scv_splits, test_size=args.scv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.num_cores, verbose=args.scv_verbose,
            )
        if args.verbose > 1:
            print('Pipeline:')
            pprint(vars(pipe))
            print('Param grid:')
            pprint(param_grid)
        split_num = 1
        split_results = []
        sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size)
        for tr_idxs, te_idxs in sss.split(X, y):
            search.fit(X[tr_idxs], y[tr_idxs])
            feature_idxs = np.arange(X[tr_idxs].shape[1])
            for step in search.best_estimator_.named_steps:
                if hasattr(search.best_estimator_.named_steps[step], 'get_support'):
                    feature_idxs = feature_idxs[search.best_estimator_.named_steps[step].get_support(indices=True)]
            feature_names = np.array(base.colnames(df_X_tr), dtype=str)[feature_idxs]
            weights = np.array([], dtype=float)
            if hasattr(search.best_estimator_.named_steps['clf'], 'coef_'):
                weights = np.square(search.best_estimator_.named_steps['clf'].coef_[0])
            elif hasattr(search.best_estimator_.named_steps['clf'], 'feature_importances_'):
                weights = search.best_estimator_.named_steps['clf'].feature_importances_
            elif (hasattr(search.best_estimator_.named_steps['fs2'], 'estimator_') and
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
                'Dataset:', dataset_tr_name,
                ' Split: %2s' % split_num,
                ' ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
                ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
                ' Features: %3s' % feature_idxs.size,
                ' Params:', search.best_params_,
            )
            if args.verbose > 1:
                if weights.size > 0:
                    feature_ranks = sorted(zip(weights, feature_idxs, feature_names), reverse=True)
                    print('Feature Rankings:')
                    for weight, _, feature in feature_ranks: print(feature, '\t', weight)
                else:
                    feature_ranks = sorted(zip(feature_idxs, feature_names), reverse=True)
                    print('Features:')
                    for _, feature in feature_ranks: print(feature)
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
        roc_aucs_cv, roc_aucs_te, bcrs_cv, bcrs_te, num_features = [], [], [], [], []
        for split_result in split_results:
            roc_aucs_cv.append(split_result['roc_auc_cv'])
            roc_aucs_te.append(split_result['roc_auc_te'])
            bcrs_cv.append(split_result['bcr_cv'])
            bcrs_te.append(split_result['bcr_te'])
            num_features.append(split_result['feature_idxs'].size)
        print(
            'Dataset:', dataset_tr_name,
            ' Mean ROC AUC (CV / Test): %.4f / %.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
            ' Mean BCR (CV / Test): %.4f / %.4f' % (np.mean(bcrs_cv), np.mean(bcrs_te)),
            ' Mean Features: %3d' % np.mean(num_features),
        )
        # calculate overall best ranked features
        feature_idxs = []
        for split_result in split_results: feature_idxs.extend(split_result['feature_idxs'])
        feature_idxs = sorted(list(set(feature_idxs)))
        feature_names = np.array(base.colnames(df_X_tr), dtype=str)[feature_idxs]
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
if args.pipe_memory: rmtree(cachedir)
