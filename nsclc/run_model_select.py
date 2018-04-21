#!/usr/bin/env python

import warnings
from argparse import ArgumentParser
from pprint import pprint
from os import path
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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFpr, SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import dump, Memory
from feature_selection import CFS, FCBF, ReliefF
import matplotlib.pyplot as plt
from matplotlib import style

# ignore QDA clf collinearity warnings
warnings.filterwarnings('ignore', category=UserWarning, message="^Variables are collinear")

# config
parser = ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--splits', type=int, default=80, help='num splits')
parser.add_argument('--test-size', type=float, default=0.3, help='test size')
parser.add_argument('--datasets-tr', type=str, nargs='+', help='datasets tr')
parser.add_argument('--datasets-te', type=str, nargs='+', help='datasets te')
parser.add_argument('--num-tr-combo', type=int, help='dataset tr num combos')
parser.add_argument('--no-addon-te', default=False, action='store_true', help='dataset te no addon')
parser.add_argument('--merge-type', type=str, nargs='+', help='dataset merge type')
parser.add_argument('--id-type', type=str, nargs='+', help='dataset ID type')
parser.add_argument('--norm-meth', type=str, nargs='+', help='preprocess/normalization method')
parser.add_argument('--bc-meth', type=str, nargs='+', help='batch effect correction method')
parser.add_argument('--fs-meth', type=str, nargs='+', help='feature selection method')
parser.add_argument('--slr-meth', type=str, nargs='+', default=['StandardScaler'], help='scaling method')
parser.add_argument('--clf-meth', type=str, nargs='+', help='classifier method')
parser.add_argument('--fs-skb-k', type=int, nargs='+', help='fs skb k select')
parser.add_argument('--fs-skb-k-max', type=int, default=30, help='fs skb k max')
parser.add_argument('--fs-fpr-p', type=float, nargs='+', help='fs fpr p-value')
parser.add_argument('--fs-sfm-thres', type=float, nargs='+', help='fs sfm threshold')
parser.add_argument('--fs-sfm-svm-c', type=float, nargs='+', help='fs sfm svm c')
parser.add_argument('--fs-rfe-svm-c', type=float, nargs='+', help='fs rfe svm c')
parser.add_argument('--fs-rfe-step', type=float, default=0.2, help='fs rfe step')
parser.add_argument('--fs-rfe-verbose', type=int, default=0, help='fs rfe verbosity')
parser.add_argument('--fs-fcbf-k', type=int, nargs='+', help='fs fcbf k select')
parser.add_argument('--fs-fcbf-k-max', type=int, default=30, help='fs fcbf k max')
parser.add_argument('--fs-rlf-k', type=int, nargs='+', help='fs rlf k select')
parser.add_argument('--fs-rlf-k-max', type=int, default=30, help='fs rlf k max')
parser.add_argument('--fs-rlf-n', type=int, nargs='+', help='fs rlf n neighbors')
parser.add_argument('--fs-rlf-n-max', type=int, default=20, help='fs rlf n neighbors max')
parser.add_argument('--fs-rank-meth', type=str, default='mean_weights', help='fs rank method')
parser.add_argument('--clf-svm-c', type=float, nargs='+', help='clf svm c')
parser.add_argument('--clf-svm-cache', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--clf-knn-n', type=int, nargs='+', help='clf knn neighbors')
parser.add_argument('--clf-knn-n-max', type=int, default=20, help='clf knn neighbors max')
parser.add_argument('--clf-knn-w', type=str, nargs='+', help='clf knn weights')
parser.add_argument('--clf-ext-n', type=int, nargs='+', help='clf ext n estimators')
parser.add_argument('--clf-ext-n-max', type=int, default=100, help='clf ext n estimators max')
parser.add_argument('--clf-ada-n', type=int, nargs='+', help='clf ada n estimators')
parser.add_argument('--clf-ada-n-max', type=int, default=200, help='clf ada n estimators max')
parser.add_argument('--clf-grb-n', type=int, nargs='+', help='clf grb n estimators')
parser.add_argument('--clf-grb-n-max', type=int, default=100, help='clf grb n estimators max')
parser.add_argument('--clf-grb-d', type=int, nargs='+', help='clf grb max depth')
parser.add_argument('--clf-grb-d-max', type=int, default=20, help='clf grb max depth max')
parser.add_argument('--gscv-splits', type=int, default=80, help='gscv splits')
parser.add_argument('--gscv-size', type=float, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--gscv-refit', type=str, default='roc_auc', help='gscv refit score function (roc_auc, bcr)')
parser.add_argument('--pipe-memory', default=False, action='store_true', help='turn on pipeline memory')
args = parser.parse_args()

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_eset_class_labels = robjects.globalenv['esetClassLabels']
r_eset_gene_symbols = robjects.globalenv['esetGeneSymbols']
r_limma_feature_score = robjects.globalenv['limmaFeatureScore']
numpy2ri.activate()
if args.pipe_memory:
    cachedir = mkdtemp()
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

# limma feature selection scoring function
def limma(X, y):
    f, pv = r_limma_feature_score(X, y)
    return np.array(f), np.array(pv)

# bcr performance metrics scoring function
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
    mi_score_func = memory.cache(mutual_info_classif)
    rfe_svm_estimator = CachedLinearSVC(class_weight='balanced')
    sfm_svm_estimator = CachedLinearSVC(penalty='l1', dual=False, class_weight='balanced')
    sfm_ext_estimator = CachedExtraTreesClassifier(class_weight='balanced')
    ada_lgr_estimator = CachedLogisticRegression(class_weight='balanced')
else:
    limma_score_func = limma
    mi_score_func = mutual_info_classif
    rfe_svm_estimator = LinearSVC(class_weight='balanced')
    sfm_svm_estimator = LinearSVC(penalty='l1', dual=False, class_weight='balanced')
    sfm_ext_estimator = ExtraTreesClassifier(class_weight='balanced')
    ada_lgr_estimator = LogisticRegression(class_weight='balanced')

gscv_scoring = { 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }

# specify elements in sort order (needed by code dealing with gridsearch cv_results)
if args.fs_skb_k:
    SKB_N_FEATURES = sorted(args.fs_skb_k)
else:
    SKB_N_FEATURES = list(range(1, args.fs_skb_k_max + 1))
if args.fs_fpr_p:
    SFP_ALPHA = sorted(args.fs_fpr_p)
else:
    SFP_ALPHA = np.logspace(-3, -2, 2)
if args.fs_sfm_thres:
    SFM_THRESHOLDS = sorted(args.fs_sfm_thres)
else:
    SFM_THRESHOLDS = np.logspace(-9, -5, 5)
if args.fs_sfm_svm_c:
    SFM_SVC_C = sorted(args.fs_sfm_svm_c)
else:
    SFM_SVC_C = np.logspace(-2, 2, 5)
if args.fs_rfe_svm_c:
    RFE_SVC_C = sorted(args.fs_rfe_svm_c)
else:
    RFE_SVC_C = np.logspace(-7, 2, 10)
if args.fs_fcbf_k:
    FCBF_N_FEATURES = sorted(args.fs_fcbf_k)
else:
    FCBF_N_FEATURES = list(range(1, args.fs_fcbf_k_max + 1))
if args.fs_rlf_k:
    RLF_N_FEATURES = sorted(args.fs_rlf_k)
else:
    RLF_N_FEATURES = list(range(1, args.fs_rlf_k_max + 1))
if args.fs_rlf_n:
    RLF_KNN_N = sorted(args.fs_rlf_n)
else:
    RLF_KNN_N = list(range(1, args.fs_rlf_n_max + 1, 1))
if args.clf_svm_c:
    CLF_SVC_C = sorted(args.clf_svm_c)
else:
    CLF_SVC_C = np.logspace(-7, 3, 11)
if args.clf_knn_n:
    CLF_KNN_N = sorted(args.clf_knn_n)
else:
    CLF_KNN_N = list(range(1, args.clf_knn_n_max + 1, 1))
if args.clf_knn_w:
    CLF_KNN_WEIGHTS = sorted(args.clf_knn_w)
else:
    CLF_KNN_WEIGHTS = ['uniform', 'distance']
if args.clf_ext_n:
    CLF_EXT_N_ESTS = sorted(args.clf_ext_n)
else:
    CLF_EXT_N_ESTS = list(range(5, args.clf_ext_n_max + 1, 5))
if args.clf_ada_n:
    CLF_ADA_N_ESTS = sorted(args.clf_ada_n)
else:
    CLF_ADA_N_ESTS = list(range(10, args.clf_ada_n_max + 1, 10))
if args.clf_grb_n:
    CLF_GRB_N_ESTS = sorted(args.clf_grb_n)
else:
    CLF_GRB_N_ESTS = list(range(5, args.clf_grb_n_max + 1, 5))
if args.clf_grb_d:
    CLF_GRB_MAX_D = sorted(args.clf_grb_d)
else:
    CLF_GRB_MAX_D = list(range(1, args.clf_grb_d_max + 1, 1))

pipeline_order = [
    'fs1',
    'slr',
    'fs2',
    'fs3',
    'clf',
]
pipelines = {
    'slr': {
        'StandardScaler': {
            'steps': [
                ('slr', StandardScaler()),
            ],
            'param_grid': [
                { },
            ],
        },
    },
    'fs': {
        'Limma-KBest': {
            'steps': [
                ('fs1', SelectKBest(limma_score_func)),
            ],
            'param_grid': [
                {
                    'fs1__k': SKB_N_FEATURES,
                },
            ],
        },
        'MI-KBest': {
            'steps': [
                ('fs2', SelectKBest(mi_score_func)),
            ],
            'param_grid': [
                {
                    'fs2__k': SKB_N_FEATURES,
                },
            ],
        },
        'SVM-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(sfm_svm_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__k': SKB_N_FEATURES,
                    'fs2__estimator__C': SFM_SVC_C,
                    'fs2__threshold': SFM_THRESHOLDS,
                },
            ],
        },
        'ExtraTrees-SFM-KBest': {
            'steps': [
                ('fs2', SelectFromModel(sfm_ext_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__k': SKB_N_FEATURES,
                    'fs2__estimator__n_estimators': CLF_EXT_N_ESTS,
                    'fs2__threshold': SFM_THRESHOLDS,
                },
            ],
        },
        'Limma-Fpr-SVM-RFE': {
            'steps': [
                ('fs1', SelectFpr(limma_score_func)),
                ('fs3', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs1__alpha': SFP_ALPHA,
                    'fs3__estimator__C': RFE_SVC_C,
                    'fs3__n_features_to_select': SKB_N_FEATURES,
                },
            ],
        },
        'SVM-SFM-RFE': {
            'steps': [
                ('fs2', SelectFromModel(sfm_svm_estimator)),
                ('fs3', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__C': SFM_SVC_C,
                    'fs2__threshold': SFM_THRESHOLDS,
                    'fs3__estimator__C': RFE_SVC_C,
                    'fs3__n_features_to_select': SKB_N_FEATURES,
                },
            ],
        },
        'ExtraTrees-SFM-RFE': {
            'steps': [
                ('fs2', SelectFromModel(sfm_ext_estimator)),
                ('fs3', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': CLF_EXT_N_ESTS,
                    'fs2__threshold': SFM_THRESHOLDS,
                    'fs3__estimator__C': RFE_SVC_C,
                    'fs3__n_features_to_select': SKB_N_FEATURES,
                },
            ],
        },
        # 'SVM-RFE': {
        #     'steps': [
        #         ('fs3', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
        #     ],
        #     'param_grid': [
        #         {
        #             'fs3__estimator__C': RFE_SVC_C,
        #             'fs3__n_features_to_select': SKB_N_FEATURES,
        #         },
        #     ],
        # },
        'Limma-KBest-FCBF': {
            'steps': [
                ('fs1', SelectKBest(limma_score_func)),
                ('fs2', FCBF()),
            ],
            'param_grid': [
                {
                    'fs1__k': SKB_N_FEATURES,
                    'fs2__k': FCBF_N_FEATURES,
                },
            ],
        },
        # 'Limma-KBest-ReliefF': {
        #     'steps': [
        #         ('fs1', SelectKBest(limma_score_func)),
        #         ('fs2', ReliefF()),
        #     ],
        #     'param_grid': [
        #         {
        #             'fs1__k': SKB_N_FEATURES,
        #             'fs2__k': RLF_N_FEATURES,
        #             'fs__n_neighbors': RLF_KNN_N,
        #         },
        #     ],
        # },
        # 'Limma-KBest-CFS': {
        #     'steps': [
        #         ('fs1', SelectKBest(limma_score_func)),
        #         ('fs2', CFS()),
        #     ],
        #     'param_grid': [
        #         {
        #             'fs1__k': SKB_N_FEATURES,
        #         },
        #     ],
        # },
    },
    'clf': {
        'LinearSVC': {
            'steps': [
                ('clf', LinearSVC(class_weight='balanced')),
            ],
            'param_grid': [
                {
                    'clf__C': CLF_SVC_C,
                },
            ],
        },
        'kNN': {
            'steps': [
                ('clf', KNeighborsClassifier()),
            ],
            'param_grid': [
                {
                    'clf__n_neighbors': CLF_KNN_N,
                    'clf__weights': CLF_KNN_WEIGHTS,
                },
            ],
        },
        'DT': {
            'steps': [
                ('clf', DecisionTreeClassifier(class_weight='balanced')),
            ],
            'param_grid': [
                { },
            ],
        },
        'ExtraTrees': {
            'steps': [
                ('clf', ExtraTreesClassifier(class_weight='balanced')),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_EXT_N_ESTS,
                },
            ],
        },
        'RandomForest': {
            'steps': [
                ('clf', RandomForestClassifier(class_weight='balanced')),
            ],
            'param_grid': [
                {
                    'clf__n_estimators': CLF_EXT_N_ESTS,
                },
            ],
        },
        'AdaBoost': {
            'steps': [
                ('clf', AdaBoostClassifier(ada_lgr_estimator)),
            ],
            'param_grid': [
                {
                    'clf__base_estimator__C': CLF_SVC_C,
                    'clf__n_estimators': CLF_ADA_N_ESTS,
                },
            ],
        },
        'GradientBoosting': {
            'steps': [
                ('clf', GradientBoostingClassifier()),
            ],
            'param_grid': [
                {
                    'clf__max_depth': CLF_GRB_MAX_D,
                    'clf__n_estimators': CLF_GRB_N_ESTS,
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
        'MLPClassifier': {
            'steps': [
                ('clf', MLPClassifier()),
            ],
            'param_grid': [
                {
                    'clf__alpha': CLF_SVC_C,
                    'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
                },
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
    },
}
dataset_names = [
    'gse8894',
    'gse30219',
    'gse31210',
    'gse37745',
    'gse50081'
]
norm_methods = [
    'gcrma',
    'rma',
    'mas5',
]
id_types = [
    'none',
    'gene',
]
merge_types = [
    'none',
    'merged',
]
bc_methods = [
    'none',
    'ctr',
    'std',
    'rta',
    'rtg',
    'qnorm',
    'cbt',
    # 'fab',
    'sva',
    'stica0',
    'stica025',
    'stica05',
    'stica1',
    'svd',
]

# analyses
if args.analysis == 1:
    norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
    prep_methods = [norm_meth]
    if args.id_type and args.id_type != 'none':
        id_type = [x for x in id_types if x in args.id_type][0]
        prep_methods.append(id_type)
    if args.merge_type and args.merge_type != 'none':
        merge_type = [x for x in merge_types if x in args.merge_type][0]
        prep_methods.append(merge_type)
    if args.bc_meth and args.bc_meth != 'none':
        bc_meth = [x for x in bc_methods if x in args.bc_meth][0]
        prep_methods.append(bc_meth)
    args.datasets_tr = natsorted(args.datasets_tr)
    dataset_name = '_'.join(args.datasets_tr + prep_methods + ['tr'])
    print('Dataset:', dataset_name)
    eset_name = 'eset_' + dataset_name
    base.load('data/' + eset_name + '.Rda')
    eset = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_name])
    X = np.array(base.t(biobase.exprs(eset)))
    y = np.array(r_eset_class_labels(eset), dtype=int)
    args.fs_meth = args.fs_meth[0]
    args.slr_meth = args.slr_meth[0]
    args.clf_meth = args.clf_meth[0]
    param_grid = []
    for fs_params in pipelines['fs'][args.fs_meth]['param_grid']:
        for slr_params in pipelines['slr'][args.slr_meth]['param_grid']:
            for clf_params in pipelines['clf'][args.clf_meth]['param_grid']:
                param_grid.append({ **fs_params, **slr_params, **clf_params })
    grid = GridSearchCV(
        Pipeline(sorted(
            pipelines['fs'][args.fs_meth]['steps'] +
            pipelines['slr'][args.slr_meth]['steps'] +
            pipelines['clf'][args.clf_meth]['steps'],
            key=lambda s: pipeline_order.index(s[0])
        ), memory=memory), param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
        error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    split_idx = 0
    split_results = []
    param_cv_scores = {}
    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size)
    for tr_idxs, te_idxs in sss.split(X, y):
        grid.fit(X[tr_idxs], y[tr_idxs])
        feature_idxs = np.arange(X[tr_idxs].shape[1])
        for step in grid.best_estimator_.named_steps:
            if hasattr(grid.best_estimator_.named_steps[step], 'get_support'):
                feature_idxs = feature_idxs[grid.best_estimator_.named_steps[step].get_support(indices=True)]
        feature_names = np.array(biobase.featureNames(eset), dtype=str)[feature_idxs]
        if hasattr(grid.best_estimator_.named_steps['clf'], 'coef_'):
            weights = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
        elif hasattr(grid.best_estimator_.named_steps['clf'], 'feature_importances_'):
            weights = grid.best_estimator_.named_steps['clf'].feature_importances_
        roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
        bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
        if hasattr(grid, 'decision_function'):
            y_score = grid.decision_function(X[te_idxs])
        else:
            y_score = grid.predict_proba(X[te_idxs])[:,1]
        roc_auc_te = roc_auc_score(y[te_idxs], y_score)
        fpr, tpr, thres = roc_curve(y[te_idxs], y_score, pos_label=1)
        y_pred = grid.predict(X[te_idxs])
        bcr_te = bcr_score(y[te_idxs], y_pred)
        feature_ranks = sorted(
            zip(weights, feature_idxs, feature_names, r_eset_gene_symbols(eset, feature_idxs + 1)),
            reverse=True,
        )
        print(
            'Split: %3s' % (split_idx + 1),
            ' ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
            ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
            ' Features: %3s' % feature_idxs.size,
            ' Params:',  grid.best_params_,
        )
        # print('Rankings:')
        # for weight, _, feature, symbol in feature_ranks: print(feature, '\t', symbol, '\t', weight)
        for param_idx, param in enumerate(param_grid[0]):
            if '__' in param and len(param_grid[0][param]) > 1:
                new_shape = (
                    len(param_grid[0][param]),
                    np.prod([len(v) for k,v in param_grid[0].items() if k != param])
                )
                if param in ('fs2__threshold'):
                    xaxis_group_sorted_idxs = np.argsort(
                        np.ma.getdata(grid.cv_results_['param_' + param]).astype(str)
                    )
                else:
                    xaxis_group_sorted_idxs = np.argsort(
                        np.ma.getdata(grid.cv_results_['param_' + param])
                    )
                if not param in param_cv_scores: param_cv_scores[param] = {}
                for metric in gscv_scoring.keys():
                    mean_scores_cv = np.reshape(
                        grid.cv_results_['mean_test_' + metric][xaxis_group_sorted_idxs], new_shape
                    )
                    std_scores_cv = np.reshape(
                        grid.cv_results_['std_test_' + metric][xaxis_group_sorted_idxs], new_shape
                    )
                    mean_scores_cv_max_idxs = np.argmax(mean_scores_cv, axis=1)
                    mean_scores_cv = mean_scores_cv[
                        np.arange(len(mean_scores_cv)), mean_scores_cv_max_idxs
                    ]
                    std_scores_cv = std_scores_cv[
                        np.arange(len(std_scores_cv)), mean_scores_cv_max_idxs
                    ]
                    if split_idx == 0:
                        param_cv_scores[param][metric] = mean_scores_cv
                    else:
                        param_cv_scores[param][metric] = np.vstack(
                            (param_cv_scores[param][metric], mean_scores_cv)
                        )
        split_results.append({
            'grid': grid,
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
        split_idx += 1
    # plot grid search parameters vs cv perf metrics
    for param_idx, param in enumerate(param_cv_scores):
        mean_roc_aucs_cv = np.mean(param_cv_scores[param]['roc_auc'], axis=0)
        mean_bcrs_cv = np.mean(param_cv_scores[param]['bcr'], axis=0)
        std_roc_aucs_cv = np.std(param_cv_scores[param]['roc_auc'], axis=0)
        std_bcrs_cv = np.std(param_cv_scores[param]['bcr'], axis=0)
        plt.figure('Figure 1-' + str(param_idx + 1))
        plt.rcParams['font.size'] = 16
        if param in (
            'fs1__k', 'fs2__k', 'fs2__estimator__n_estimators', 'fs3__n_features_to_select',
        ):
            x_axis = param_grid[0][param]
            plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
            plt.xticks(x_axis)
        elif param in (
            'fs1__alpha', 'fs2__estimator__C', 'fs2__threshold', 'fs3__estimator__C', 'clf__C',
        ):
            x_axis = range(len(param_grid[0][param]))
            plt.xticks(x_axis, param_grid[0][param])
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
    roc_aucs_cv, roc_aucs_te, bcrs_cv, bcrs_te, num_features = [], [], [], [], []
    for split_result in split_results:
        roc_aucs_cv.append(split_result['roc_auc_cv'])
        roc_aucs_te.append(split_result['roc_auc_te'])
        bcrs_cv.append(split_result['bcr_cv'])
        bcrs_te.append(split_result['bcr_te'])
        num_features.append(split_result['feature_idxs'].size)
    print(
        'Mean ROC AUC (CV / Test): %.4f / %.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
        ' Mean BCR (CV / Test): %.4f / %.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
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
            roc_auc_mx[feature_mx_idx[feature_idx]][split_idx] = split_result['roc_auc_te']
            bcr_mx[feature_mx_idx[feature_idx]][split_idx] = split_result['bcr_te']
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
    else:
        feature_ranks = feature_mean_bcrs
    print('Rankings:')
    for rank, feature, symbol in sorted(
        zip(feature_ranks, feature_names, r_eset_gene_symbols(eset, np.array(feature_idxs) + 1)),
        reverse=True
    ): print(feature, '\t', symbol, '\t', rank)
elif args.analysis == 2:
    norm_meth = [x for x in norm_methods if x in args.norm_meth][0]
    prep_methods = [norm_meth]
    if args.id_type and args.id_type != 'none':
        id_type = [x for x in id_types if x in args.id_type][0]
        prep_methods.append(id_type)
    if args.merge_type and args.merge_type != 'none':
        merge_type = [x for x in merge_types if x in args.merge_type][0]
        prep_methods.append(merge_type)
    if args.bc_meth and args.bc_meth != 'none':
        bc_meth = [x for x in bc_methods if x in args.bc_meth][0]
        prep_methods.append(bc_meth)
    args.datasets_tr = natsorted(args.datasets_tr)
    dataset_tr_name = '_'.join(args.datasets_tr + prep_methods + ['tr'])
    print('Train Dataset:', dataset_tr_name)
    eset_tr_name = 'eset_' + dataset_tr_name
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
    args.fs_meth = args.fs_meth[0]
    args.slr_meth = args.slr_meth[0]
    args.clf_meth = args.clf_meth[0]
    param_grid = []
    for fs_params in pipelines['fs'][args.fs_meth]['param_grid']:
        for slr_params in pipelines['slr'][args.slr_meth]['param_grid']:
            for clf_params in pipelines['clf'][args.clf_meth]['param_grid']:
                param_grid.append({ **fs_params, **slr_params, **clf_params })
    grid = GridSearchCV(
        Pipeline(sorted(
            pipelines['fs'][args.fs_meth]['steps'] +
            pipelines['slr'][args.slr_meth]['steps'] +
            pipelines['clf'][args.clf_meth]['steps'],
            key=lambda s: pipeline_order.index(s[0])
        ), memory=memory), param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
        error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X_tr, y_tr)
    dump(grid, '_'.join([
        'results/grid', dataset_tr_name, args.slr_meth.lower(), args.fs_meth.lower(), args.clf_meth.lower()
    ]) + '.pkl')
    feature_idxs = np.arange(X_tr.shape[1])
    for step in grid.best_estimator_.named_steps:
        if hasattr(grid.best_estimator_.named_steps[step], 'get_support'):
            feature_idxs = feature_idxs[grid.best_estimator_.named_steps[step].get_support(indices=True)]
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)[feature_idxs]
    if hasattr(grid.best_estimator_.named_steps['clf'], 'coef_'):
        weights = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    elif hasattr(grid.best_estimator_.named_steps['clf'], 'feature_importances_'):
        weights = grid.best_estimator_.named_steps['clf'].feature_importances_
    roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
    bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
    feature_ranks = sorted(
        zip(weights, feature_idxs, feature_names, r_eset_gene_symbols(eset_tr, feature_idxs + 1)),
        reverse=True,
    )
    print(
        'ROC AUC (CV): %.4f' % roc_auc_cv,
        ' BCR (CV): %.4f' % bcr_cv,
        ' Features: %3s' % feature_idxs.size,
        ' Params:',  grid.best_params_,
    )
    print('Rankings:')
    for weight, _, feature, symbol in feature_ranks: print(feature, '\t', symbol, '\t', weight)
    # pprint(grid.cv_results_)
    # pprint(param_grid)
    # plot grid search parameters vs cv perf metrics
    for param_idx, param in enumerate(param_grid[0]):
        if '__' in param and len(param_grid[0][param]) > 1:
            new_shape = (
                len(param_grid[0][param]),
                np.prod([len(v) for k,v in param_grid[0].items() if k != param])
            )
            if param in ('fs2__threshold'):
                xaxis_group_sorted_idxs = np.argsort(
                    np.ma.getdata(grid.cv_results_['param_' + param]).astype(str)
                )
            else:
                xaxis_group_sorted_idxs = np.argsort(
                    np.ma.getdata(grid.cv_results_['param_' + param])
                )
            plt.figure('Figure 2-' + str(param_idx + 1))
            plt.rcParams['font.size'] = 16
            if param in (
                'fs1__k', 'fs2__k', 'fs2__estimator__n_estimators', 'fs3__n_features_to_select',
            ):
                x_axis = param_grid[0][param]
                plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
                plt.xticks(x_axis)
            elif param in (
                'fs1__alpha', 'fs2__estimator__C', 'fs2__threshold', 'fs3__estimator__C', 'clf__C',
            ):
                x_axis = range(len(param_grid[0][param]))
                plt.xticks(x_axis, param_grid[0][param])
            plt.title(
                dataset_tr_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
                'Effect of ' + param + ' on CV Performance Metrics'
            )
            plt.xlabel(param)
            plt.ylabel('CV Score')
            for metric_idx, metric in enumerate(sorted(gscv_scoring.keys(), reverse=True)):
                mean_scores_cv = np.reshape(
                    grid.cv_results_['mean_test_' + metric][xaxis_group_sorted_idxs], new_shape
                )
                std_scores_cv = np.reshape(
                    grid.cv_results_['std_test_' + metric][xaxis_group_sorted_idxs], new_shape
                )
                mean_scores_cv_max_idxs = np.argmax(mean_scores_cv, axis=1)
                mean_scores_cv = mean_scores_cv[
                    np.arange(len(mean_scores_cv)), mean_scores_cv_max_idxs
                ]
                std_scores_cv = std_scores_cv[
                    np.arange(len(std_scores_cv)), mean_scores_cv_max_idxs
                ]
                plt.plot(
                    x_axis,
                    mean_scores_cv,
                    lw=2, alpha=0.8, label='Mean ' + metric.replace('_', ' ').upper()
                )
                plt.fill_between(
                    x_axis,
                    [m - s for m, s in zip(mean_scores_cv, std_scores_cv)],
                    [m + s for m, s in zip(mean_scores_cv, std_scores_cv)],
                    color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
                )
            plt.legend(loc='lower right', fontsize='small')
            plt.grid('on')
    # plot num top-ranked features selected vs test dataset perf metrics
    plt.figure('Figure 3')
    plt.rcParams['font.size'] = 16
    plt.title(
        dataset_tr_name + ' ' + args.clf_meth + ' Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected Performance Metrics'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('Test Score')
    x_axis = range(1, feature_idxs.size + 1)
    plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
    plt.xticks(x_axis)
    ranked_feature_idxs = [x for _, x, _, _ in feature_ranks]
    pipe = Pipeline(
        pipelines['slr'][args.slr_meth]['steps'] +
        pipelines['clf'][args.clf_meth]['steps']
    )
    pipe.set_params(
        **{ k: v for k, v in grid.best_params_.items() if k.startswith('slr') or k.startswith('clf') }
    )
    for dataset_te_basename in natsorted(list(set(dataset_names) - set(args.datasets_tr))):
        if args.no_addon_te:
            dataset_te_name = '_'.join([dataset_te_basename, prep_methods[0]])
        else:
            dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
        eset_te_name = 'eset_' + dataset_te_name
        eset_te_file = 'data/' + eset_te_name + '.Rda'
        if not path.isfile(eset_te_file): continue
        base.load(eset_te_file)
        eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        X_te = np.array(base.t(biobase.exprs(eset_te)))
        y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
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
            x_axis, roc_aucs_te,
            lw=2, alpha=0.8, label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, BCR = %0.4f $\pm$ %0.2f)' % (
                dataset_te_name,
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(bcrs_te), np.std(bcrs_te),
            ),
        )
        # plt.plot(
        #     x_axis, bcrs_te,
        #     lw=2, alpha=0.8,
        # )
        # print summary info
        print(
            'Test Dataset: %3s' % dataset_te_name,
            ' ROC AUC: %.4f' % np.max(roc_aucs_te),
            ' BCR: %.4f' % np.max(bcrs_te),
        )
    plt.legend(loc='lower right', fontsize='small')
    plt.grid('on')
elif args.analysis == 3:
    prep_groups = []
    if args.norm_meth:
        norm_methods = [x for x in norm_methods if x in args.norm_meth]
    if args.id_type:
        id_types = [x for x in id_types if x in args.id_type]
    if args.merge_type:
        merge_types = [x for x in merge_types if x in args.merge_type]
    if args.bc_meth:
        bc_methods = [x for x in bc_methods if x in args.bc_meth]
    for norm_meth in norm_methods:
        for id_type in id_types:
            for merge_type in merge_types:
                for bc_meth in bc_methods:
                    prep_groups.append([
                        x for x in [norm_meth, id_type, merge_type, bc_meth] if x != 'none'
                    ])
    if (args.fs_meth and len(args.fs_meth) == 1 and
        args.slr_meth and len(args.slr_meth) == 1 and
        args.clf_meth and len(args.clf_meth) == 1):
        analysis_type = 'prep_methods'
        args.slr_meth = args.slr_meth[0]
        args.fs_meth = args.fs_meth[0]
        args.clf_meth = args.clf_meth[0]
        param_grid = []
        for fs_params in pipelines['fs'][args.fs_meth]['param_grid']:
            for slr_params in pipelines['slr'][args.slr_meth]['param_grid']:
                for clf_params in pipelines['clf'][args.clf_meth]['param_grid']:
                    param_grid.append({ **fs_params, **slr_params, **clf_params })
        grid = GridSearchCV(
            Pipeline(sorted(
                pipelines['fs'][args.fs_meth]['steps'] +
                pipelines['slr'][args.slr_meth]['steps'] +
                pipelines['clf'][args.clf_meth]['steps'],
                key=lambda s: pipeline_order.index(s[0])
            ), memory=memory), param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
        )
    else:
        analysis_type = 'all_methods'
        if args.fs_meth:
            pipelines['fs'] = { k: v for k, v in pipelines['fs'].items() if k in args.fs_meth }
        if args.slr_meth:
            pipelines['slr'] = { k: v for k, v in pipelines['slr'].items() if k in args.slr_meth }
        if args.clf_meth:
            pipelines['clf'] = { k: v for k, v in pipelines['clf'].items() if k in args.clf_meth }
        param_grid_idx = 0
        param_grid, param_grid_data = [], []
        for fs_idx, fs_meth in enumerate(pipelines['fs']):
            for fs_params in pipelines['fs'][fs_meth]['param_grid']:
                for slr_idx, slr_meth in enumerate(pipelines['slr']):
                    for slr_params in pipelines['slr'][slr_meth]['param_grid']:
                        for clf_idx, clf_meth in enumerate(pipelines['clf']):
                            for clf_params in pipelines['clf'][clf_meth]['param_grid']:
                                params = { **fs_params, **slr_params, **clf_params }
                                for (step, object) in \
                                    pipelines['fs'][fs_meth]['steps'] + \
                                    pipelines['slr'][slr_meth]['steps'] + \
                                    pipelines['clf'][clf_meth]['steps'] \
                                : params[step] = [ object ]
                                param_grid.append(params)
                                params_data = {
                                    'meth_idxs': {
                                        'fs': fs_idx, 'slr': slr_idx, 'clf': clf_idx, 'pr': 0
                                    },
                                    'grid_idxs': [],
                                }
                                for param_combo in ParameterGrid(params):
                                    params_data['grid_idxs'].append(param_grid_idx)
                                    param_grid_idx += 1
                                param_grid_data.append(params_data)
        grid = GridSearchCV(
            Pipeline(list(map(lambda x: (x, None), pipeline_order)), memory=memory),
            param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
            cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
            error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
        )
    if args.datasets_tr and args.num_tr_combo:
        dataset_tr_combos = [list(x) for x in combinations(natsorted(args.datasets_tr), args.num_tr_combo)]
    elif args.datasets_tr:
        dataset_tr_combos = [x for x in natsorted(dataset_names) if x in args.datasets_tr]
    else:
        dataset_tr_combos = [list(x) for x in combinations(dataset_names, args.num_tr_combo)]
    if args.datasets_te:
        dataset_te_basenames = [x for x in natsorted(dataset_names) if x in args.datasets_te]
    else:
        dataset_te_basenames = dataset_names
    # determine which data combinations will be used
    dataset_tr_combos_subset, dataset_te_basenames_subset, prep_groups_subset = [], [], []
    for dataset_tr_combo in dataset_tr_combos:
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        for dataset_te_basename in dataset_te_basenames:
            for prep_steps in prep_groups:
                prep_method = '_'.join(prep_steps)
                dataset_tr_name = '_'.join([dataset_tr_basename, prep_method, 'tr'])
                if args.no_addon_te:
                    dataset_te_name = '_'.join([dataset_te_basename, prep_steps[0]])
                else:
                    dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
                eset_tr_file = 'data/' + eset_tr_name + '.Rda'
                eset_te_file = 'data/' + eset_te_name + '.Rda'
                if not path.isfile(eset_tr_file) or not path.isfile(eset_te_file): continue
                dataset_tr_combos_subset.append(dataset_tr_combo)
                dataset_te_basenames_subset.append(dataset_te_basename)
                prep_groups_subset.append(prep_steps)
    dataset_tr_combos = [x for x in dataset_tr_combos if x in dataset_tr_combos_subset]
    dataset_te_basenames = [x for x in dataset_te_basenames if x in dataset_te_basenames_subset]
    prep_groups = [x for x in prep_groups if x in prep_groups_subset]
    score_dtypes = [
        ('roc_auc_cv', float), ('bcr_cv', float),
        ('roc_auc_te', float), ('bcr_te', float),
        ('num_features', int),
    ]
    te_pr_results = np.zeros((len(dataset_te_basenames), len(prep_groups)), dtype=[
        ('tr', score_dtypes, (len(dataset_tr_combos),))
    ])
    tr_pr_results = np.zeros((len(dataset_tr_combos), len(prep_groups)), dtype=[
        ('te', score_dtypes, (len(dataset_te_basenames),))
    ])
    te_fs_results = np.zeros((len(dataset_te_basenames), len(pipelines['fs'])), dtype=[
        ('tr', [ ('pr', score_dtypes, (len(prep_groups),)) ], (len(dataset_tr_combos),)),
        ('pr', [ ('tr', score_dtypes, (len(dataset_tr_combos),)) ], (len(prep_groups),))
    ])
    tr_fs_results = np.zeros((len(dataset_tr_combos), len(pipelines['fs'])), dtype=[
        ('te', [ ('pr', score_dtypes, (len(prep_groups),)) ], (len(dataset_te_basenames),)),
        ('pr', [ ('te', score_dtypes, (len(dataset_te_basenames),)) ], (len(prep_groups),))
    ])
    te_clf_results = np.zeros((len(dataset_te_basenames), len(pipelines['clf'])), dtype=[
        ('tr', [ ('pr', score_dtypes, (len(prep_groups),)) ], (len(dataset_tr_combos),)),
        ('pr', [ ('tr', score_dtypes, (len(dataset_tr_combos),)) ], (len(prep_groups),))
    ])
    tr_clf_results = np.zeros((len(dataset_tr_combos), len(pipelines['clf'])), dtype=[
        ('te', [ ('pr', score_dtypes, (len(prep_groups),)) ], (len(dataset_te_basenames),)),
        ('pr', [ ('te', score_dtypes, (len(dataset_te_basenames),)) ], (len(prep_groups),))
    ])
    pr_fs_results = np.zeros((len(prep_groups), len(pipelines['fs'])), dtype=[
        ('te', [ ('tr', score_dtypes, (len(dataset_tr_combos),)) ], (len(dataset_te_basenames),)),
        ('tr', [ ('te', score_dtypes, (len(dataset_te_basenames),)) ], (len(dataset_tr_combos),))
    ])
    pr_clf_results = np.zeros((len(prep_groups), len(pipelines['clf'])), dtype=[
        ('te', [ ('tr', score_dtypes, (len(dataset_tr_combos),)) ], (len(dataset_te_basenames),)),
        ('tr', [ ('te', score_dtypes, (len(dataset_te_basenames),)) ], (len(dataset_tr_combos),))
    ])
    for tr_idx, dataset_tr_combo in enumerate(dataset_tr_combos):
        dataset_tr_basename = '_'.join(dataset_tr_combo)
        for te_idx, dataset_te_basename in enumerate(dataset_te_basenames):
            for pr_idx, prep_steps in enumerate(prep_groups):
                prep_method = '_'.join(prep_steps)
                dataset_tr_name = '_'.join([dataset_tr_basename, prep_method, 'tr'])
                if args.no_addon_te:
                    dataset_te_name = '_'.join([dataset_te_basename, prep_steps[0]])
                else:
                    dataset_te_name = '_'.join([dataset_tr_name, dataset_te_basename, 'te'])
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
                eset_tr_file = 'data/' + eset_tr_name + '.Rda'
                eset_te_file = 'data/' + eset_te_name + '.Rda'
                if not path.isfile(eset_tr_file) or not path.isfile(eset_te_file): continue
                print(dataset_tr_name, '->', dataset_te_name)
                base.load('data/' + eset_tr_name + '.Rda')
                eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
                X_tr = np.array(base.t(biobase.exprs(eset_tr)))
                y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
                base.load('data/' + eset_te_name + '.Rda')
                eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
                X_te = np.array(base.t(biobase.exprs(eset_te)))
                y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
                grid.fit(X_tr, y_tr)
                if analysis_type == 'prep_methods':
                    dump(grid, '_'.join([
                        'results/grid', dataset_tr_name, args.slr_meth.lower(),
                         args.fs_meth.lower(), args.clf_meth.lower()
                    ]) + '.pkl')
                    feature_idxs = np.arange(X_tr.shape[1])
                    for step in grid.best_estimator_.named_steps:
                        if hasattr(grid.best_estimator_.named_steps[step], 'get_support'):
                            feature_idxs = feature_idxs[grid.best_estimator_.named_steps[step].get_support(indices=True)]
                    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)[feature_idxs]
                    if hasattr(grid.best_estimator_.named_steps['clf'], 'coef_'):
                        weights = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
                    elif hasattr(grid.best_estimator_.named_steps['clf'], 'feature_importances_'):
                        weights = grid.best_estimator_.named_steps['clf'].feature_importances_
                    roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
                    bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
                    if hasattr(grid, 'decision_function'):
                        y_score = grid.decision_function(X_te)
                    else:
                        y_score = grid.predict_proba(X_te)[:,1]
                    roc_auc_te = roc_auc_score(y_te, y_score)
                    y_pred = grid.predict(X_te)
                    bcr_te = bcr_score(y_te, y_pred)
                    print(
                        'ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
                        ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
                        ' Features: %3s' % feature_idxs.size,
                        ' Params:',  grid.best_params_,
                    )
                    # print('Rankings:')
                    # for rank, feature, symbol in sorted(
                    #     zip(weights, feature_names, r_eset_gene_symbols(eset_tr, feature_idxs + 1)),
                    #     reverse=True,
                    # ): print(feature, '\t', symbol, '\t', rank)
                    te_pr_results[te_idx, pr_idx]['tr'][tr_idx]['roc_auc_cv'] = roc_auc_cv
                    te_pr_results[te_idx, pr_idx]['tr'][tr_idx]['bcr_te'] = bcr_te
                    te_pr_results[te_idx, pr_idx]['tr'][tr_idx]['roc_auc_te'] = roc_auc_te
                    te_pr_results[te_idx, pr_idx]['tr'][tr_idx]['bcr_cv'] = bcr_cv
                    te_pr_results[te_idx, pr_idx]['tr'][tr_idx]['num_features'] = feature_idxs.size
                    tr_pr_results[tr_idx, pr_idx]['te'][te_idx] = te_pr_results[te_idx, pr_idx]['tr'][tr_idx]
                elif analysis_type == 'all_methods':
                    best_grid_idxs = []
                    meth_scores = { 'cv': {}, 'te': {} }
                    for group_idx, param_grid_group in enumerate(param_grid_data):
                        for grid_idx in param_grid_group['grid_idxs']:
                            if group_idx < len(best_grid_idxs):
                                if (grid.cv_results_['rank_test_' + args.gscv_refit][grid_idx] <
                                    grid.cv_results_['rank_test_' + args.gscv_refit][best_grid_idxs[group_idx]]):
                                    best_grid_idxs[group_idx] = grid_idx
                            else:
                                best_grid_idxs.append(grid_idx)
                        for meth_type, meth_idx in param_grid_group['meth_idxs'].items():
                            if meth_type not in meth_scores['cv']:
                                meth_scores['cv'][meth_type] = []
                            if meth_idx >= len(meth_scores['cv'][meth_type]):
                                meth_scores['cv'][meth_type].append({})
                            for metric in gscv_scoring.keys():
                                if metric not in meth_scores['cv'][meth_type][meth_idx]:
                                    meth_scores['cv'][meth_type][meth_idx][metric] = []
                                meth_scores['cv'][meth_type][meth_idx][metric].append(
                                    grid.cv_results_['mean_test_' + metric][best_grid_idxs[group_idx]]
                                )
                    for group_idx, param_grid_group in enumerate(param_grid_data):
                        params = grid.cv_results_['params'][best_grid_idxs[group_idx]]
                        pipe_steps = sorted([
                            (k, v) for k, v in params.items() if k in pipeline_order
                        ], key=lambda s: pipeline_order.index(s[0]))
                        pipe = Pipeline(pipe_steps)
                        pipe.set_params(**{ k: v for k, v in params.items() if '__' in k })
                        pipe.fit(X_tr, y_tr)
                        if hasattr(pipe, 'decision_function'):
                            y_score = pipe.decision_function(X_te)
                        else:
                            y_score = pipe.predict_proba(X_te)[:,1]
                        roc_auc_te = roc_auc_score(y_te, y_score)
                        y_pred = pipe.predict(X_te)
                        bcr_te = bcr_score(y_te, y_pred)
                        for meth_type, meth_idx in param_grid_group['meth_idxs'].items():
                            if meth_type not in meth_scores['te']:
                                meth_scores['te'][meth_type] = []
                            if meth_idx >= len(meth_scores['te'][meth_type]):
                                meth_scores['te'][meth_type].append({ 'roc_auc': [], 'bcr': [] })
                            meth_scores['te'][meth_type][meth_idx]['roc_auc'].append(roc_auc_te)
                            meth_scores['te'][meth_type][meth_idx]['bcr'].append(bcr_te)
                    for category, category_scores in meth_scores.items():
                        for meth_type, meth_type_scores in category_scores.items():
                            for meth_idx, meth_metric_scores in enumerate(meth_type_scores):
                                for metric, metric_scores in meth_metric_scores.items():
                                    field_key = metric + '_' + category
                                    mean_score = np.mean(metric_scores)
                                    if meth_type == 'fs':
                                        te_fs_results[te_idx, meth_idx]['tr'][tr_idx]['pr'][pr_idx][field_key] = mean_score
                                        te_fs_results[te_idx, meth_idx]['pr'][pr_idx]['tr'][tr_idx][field_key] = mean_score
                                        tr_fs_results[tr_idx, meth_idx]['te'][te_idx]['pr'][pr_idx][field_key] = mean_score
                                        tr_fs_results[tr_idx, meth_idx]['pr'][pr_idx]['te'][te_idx][field_key] = mean_score
                                        pr_fs_results[pr_idx, meth_idx]['tr'][tr_idx]['te'][te_idx][field_key] = mean_score
                                        pr_fs_results[pr_idx, meth_idx]['te'][te_idx]['tr'][tr_idx][field_key] = mean_score
                                    elif meth_type == 'clf':
                                        te_clf_results[te_idx, meth_idx]['tr'][tr_idx]['pr'][pr_idx][field_key] = mean_score
                                        te_clf_results[te_idx, meth_idx]['pr'][pr_idx]['tr'][tr_idx][field_key] = mean_score
                                        tr_clf_results[tr_idx, meth_idx]['te'][te_idx]['pr'][pr_idx][field_key] = mean_score
                                        tr_clf_results[tr_idx, meth_idx]['pr'][pr_idx]['te'][te_idx][field_key] = mean_score
                                        pr_clf_results[pr_idx, meth_idx]['tr'][tr_idx]['te'][te_idx][field_key] = mean_score
                                        pr_clf_results[pr_idx, meth_idx]['te'][te_idx]['tr'][tr_idx][field_key] = mean_score
                                    elif meth_type == 'pr':
                                        te_pr_results[te_idx, pr_idx]['tr'][tr_idx][field_key] = mean_score
                                        tr_pr_results[tr_idx, pr_idx]['te'][te_idx][field_key] = mean_score
                base.remove(eset_tr_name)
                base.remove(eset_te_name)
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
        # plot te_pr_results
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': te_pr_results,
            'row_names': dataset_te_basenames,
            'col_results_key': 'tr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': te_pr_results.T,
            'row_names': prep_methods,
            'col_results_key': 'tr',
        },
        # plot tr_pr_results
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': tr_pr_results,
            'row_names': dataset_tr_basenames,
            'col_results_key': 'te',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': tr_pr_results.T,
            'row_names': prep_methods,
            'col_results_key': 'te',
        },
        # plot te_fs_results
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': te_fs_results,
            'row_names': dataset_te_basenames,
            'col_results_key': 'tr',
            'sub_results_key': 'pr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': te_fs_results.T,
            'row_names': list(pipelines['fs'].keys()),
            'col_results_key': 'tr',
            'sub_results_key': 'pr',
        },
        # plot tr_fs_results
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': tr_fs_results,
            'row_names': dataset_tr_basenames,
            'col_results_key': 'te',
            'sub_results_key': 'pr',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': tr_fs_results.T,
            'row_names': list(pipelines['fs'].keys()),
            'col_results_key': 'te',
            'sub_results_key': 'pr',
        },
        # plot te_clf_results
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Test Dataset',
            'title_sub': title_sub,
            'results': te_clf_results,
            'row_names': dataset_te_basenames,
            'col_results_key': 'tr',
            'sub_results_key': 'pr',
        },
        {
            'x_axis': range(1, len(dataset_te_basenames) + 1),
            'x_axis_labels': dataset_te_basenames,
            'x_axis_title': 'Test Dataset',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': te_clf_results.T,
            'row_names': list(pipelines['clf'].keys()),
            'col_results_key': 'tr',
            'sub_results_key': 'pr',
        },
        # plot tr_clf_results
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Train Dataset',
            'title_sub': title_sub,
            'results': tr_clf_results,
            'row_names': dataset_tr_basenames,
            'col_results_key': 'te',
            'sub_results_key': 'pr',
        },
        {
            'x_axis': range(1, len(dataset_tr_basenames) + 1),
            'x_axis_labels': dataset_tr_basenames,
            'x_axis_title': 'Train Dataset',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': tr_clf_results.T,
            'row_names': list(pipelines['clf'].keys()),
            'col_results_key': 'te',
            'sub_results_key': 'pr',
        },
        # plot pr_fs_results
        {
            'x_axis': range(1, len(list(pipelines['fs'].keys())) + 1),
            'x_axis_labels': list(pipelines['fs'].keys()),
            'x_axis_title': 'Feature Selection Method',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': pr_fs_results,
            'row_names': prep_methods,
            'col_results_key': 'te',
            'sub_results_key': 'tr',
        },
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Feature Selection Method',
            'title_sub': title_sub,
            'results': pr_fs_results.T,
            'row_names': list(pipelines['fs'].keys()),
            'col_results_key': 'te',
            'sub_results_key': 'tr',
        },
        # plot pr_clf_results
        {
            'x_axis': range(1, len(list(pipelines['clf'].keys())) + 1),
            'x_axis_labels': list(pipelines['clf'].keys()),
            'x_axis_title': 'Classifier Algorithm',
            'lines_title': 'Preprocessing Method',
            'title_sub': title_sub,
            'results': pr_clf_results,
            'row_names': prep_methods,
            'col_results_key': 'te',
            'sub_results_key': 'tr',
        },
        {
            'x_axis': range(1, len(prep_methods) + 1),
            'x_axis_labels': prep_methods,
            'x_axis_title': 'Preprocessing Method',
            'lines_title': 'Classifier Algorithm',
            'title_sub': title_sub,
            'results': pr_clf_results.T,
            'row_names': list(pipelines['clf'].keys()),
            'col_results_key': 'te',
            'sub_results_key': 'tr',
        },
    ]
    for figure_idx, figure in enumerate(figures):
        figure_num = figure_idx + 3
        for metric_idx, metric in enumerate(sorted(gscv_scoring.keys(), reverse=True)):
            metric_title = metric.replace('_', ' ').upper()
            plt.figure('Figure ' + str(figure_num) + '-' + str(metric_idx + 1))
            plt.rcParams['font.size'] = 14
            plt.title(
                'Effect of ' + figure['x_axis_title'] + ' on ' +
                metric_title + ' for each ' + figure['lines_title'] + '\n' +
                figure['title_sub']
            )
            plt.xlabel(figure['x_axis_title'])
            plt.ylabel(metric_title)
            plt.xticks(figure['x_axis'], figure['x_axis_labels'])
            for row_idx, row_results in enumerate(figure['results']):
                mean_scores_cv = np.full((figure['results'].shape[1],), np.nan, dtype=float)
                range_scores_cv = np.full((2, figure['results'].shape[1]), np.nan, dtype=float)
                mean_scores_te = np.full((figure['results'].shape[1],), np.nan, dtype=float)
                range_scores_te = np.full((2, figure['results'].shape[1]), np.nan, dtype=float)
                num_features = np.array([], dtype=int)
                for col_idx, col_results in enumerate(row_results):
                    scores_cv, scores_te = [], []
                    for result in col_results[figure['col_results_key']]:
                        result_num_features = []
                        if 'sub_results_key' in figure and figure['sub_results_key'] in result.dtype.fields:
                            sub_scores_cv, sub_scores_te = [], []
                            for sub_result in result[figure['sub_results_key']]:
                                sub_scores_cv.append(sub_result[metric + '_cv'])
                                sub_scores_te.append(sub_result[metric + '_te'])
                                result_num_features.append(sub_result['num_features'])
                            score_cv = np.mean(sub_scores_cv)
                            score_te = np.mean(sub_scores_te)
                        else:
                            score_cv = result[metric + '_cv']
                            score_te = result[metric + '_te']
                            result_num_features.append(result['num_features'])
                        if score_cv == 0: continue
                        scores_cv.append(score_cv)
                        scores_te.append(score_te)
                        num_features = np.append(num_features, result_num_features)
                    if scores_cv:
                        mean_scores_cv[col_idx] = np.mean(scores_cv)
                        range_scores_cv[0][col_idx] = np.mean(scores_cv) - min(scores_cv)
                        range_scores_cv[1][col_idx] = max(scores_cv) - np.mean(scores_cv)
                        mean_scores_te[col_idx] = np.mean(scores_te)
                        range_scores_te[0][col_idx] = np.mean(scores_te) - min(scores_te)
                        range_scores_te[1][col_idx] = max(scores_te) - np.mean(scores_te)
                if not np.all(np.isnan(mean_scores_cv)):
                    label_values = (
                        figure['row_names'][row_idx],
                        np.mean(mean_scores_cv[~np.isnan(mean_scores_cv)]),
                        np.std(mean_scores_cv[~np.isnan(mean_scores_cv)]),
                        np.mean(mean_scores_te[~np.isnan(mean_scores_te)]),
                        np.std(mean_scores_te[~np.isnan(mean_scores_te)]),
                    )
                    if np.mean(num_features) == 0:
                        label = r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f)'
                    elif np.std(num_features) == 0:
                        label = r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d)'
                        label_values = (label_values, np.mean(num_features))
                    else:
                        label = r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)'
                        label_values = (label_values, np.mean(num_features), np.std(num_features))
                    color = next(plt.gca()._get_lines.prop_cycler)['color']
                    plt.errorbar(
                        figure['x_axis'], mean_scores_cv, yerr=range_scores_cv, lw=2, alpha=0.8,
                        linestyle='--', capsize=15, elinewidth=2, markeredgewidth=2, marker='s', color=color,
                    )
                    plt.errorbar(
                        figure['x_axis'], mean_scores_te, yerr=range_scores_te, lw=2, alpha=0.8,
                        capsize=15, elinewidth=2, markeredgewidth=2, marker='s', color=color,
                        label=label % label_values,
                    )
            plt.legend(loc='best', fontsize='x-small')
            plt.grid('on')

plt.show()
if args.pipe_memory: rmtree(cachedir)
