#!/usr/bin/env python

from argparse import ArgumentParser
from pprint import pprint
from os import path
from tempfile import mkdtemp
from shutil import rmtree
from natsort import natsorted
import numpy as np
import rpy2.rinterface as rinterface
rinterface.set_initoptions((b'rpy2', b'--quiet', b'--no-save', b'--max-ppsize=500000'))
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
# from rpy2.robjects import pandas2ri
# import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFpr, SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import dump, Memory
from feature_selection import CFS, FCBF, ReliefF
import matplotlib.pyplot as plt
from matplotlib import style

# config
parser = ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--splits', type=int, default=80, help='num splits')
parser.add_argument('--test-size', type=float, default=0.3, help='test size')
parser.add_argument('--dataset-tr-cmb', type=int, help='dataset tr num combos')
parser.add_argument('--datasets-tr', type=str, nargs='+', help='datasets tr')
parser.add_argument('--datasets-te', type=str, nargs='+', help='datasets te')
parser.add_argument('--norm-meth', type=str, help='preprocess/normalization method')
parser.add_argument('--bc-meth', type=str, help='batch effect correction method')
parser.add_argument('--fs-meth', type=str, help='feature selection method')
parser.add_argument('--slr-meth', type=str, default='Standard', help='scaling method')
parser.add_argument('--clf-meth', type=str, default='LinearSVC', help='classifier method')
parser.add_argument('--no-addon-te', default=False, action='store_true', help='dataset te no addon')
parser.add_argument('--fs-skb-k', type=int, nargs='+', help='fs skb k select')
parser.add_argument('--fs-skb-k-max', type=int, default=30, help='fs skb k max')
parser.add_argument('--fs-fcbf-k', type=int, nargs='+', help='fs fcbf k select')
parser.add_argument('--fs-fcbf-k-max', type=int, default=30, help='fs fcbf k max')
parser.add_argument('--fs-fpr-p', type=float, nargs='+', help='fs fpr p-value')
parser.add_argument('--fs-sfm-thres', type=float, nargs='+', help='fs sfm threshold')
parser.add_argument('--fs-sfm-c', type=float, nargs='+', help='fs sfm svm c')
parser.add_argument('--fs-sfm-e', type=int, nargs='+', help='fs sfm ext estimators')
parser.add_argument('--fs-sfm-e-max', type=int, default=100, help='fs sfm ext estimators max')
parser.add_argument('--fs-rfe-n', type=int, nargs='+', help='fs rfe n select')
parser.add_argument('--fs-rfe-n-max', type=int, default=30, help='fs rfe n max')
parser.add_argument('--fs-rfe-c', type=float, nargs='+', help='fs rfe c')
parser.add_argument('--fs-rfe-step', type=float, default=0.2, help='fs rfe step')
parser.add_argument('--fs-rfe-verbose', type=int, default=0, help='fs rfe verbosity')
parser.add_argument('--fs-rank-meth', type=str, default='mean_weights', help='fs rank method (mean_weights, mean_roc_aucs, mean_bcrs)')
parser.add_argument('--clf-svm-c', type=float, nargs='+', help='clf svm c')
parser.add_argument('--clf-svm-cache', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--gscv-splits', type=int, default=80, help='gscv splits')
parser.add_argument('--gscv-size', type=float, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--gscv-refit', type=str, default='roc_auc', help='gscv refit score function (roc_auc, bcr)')
parser.add_argument('--gscv-no-memory', default=False, action='store_true', help='gscv no pipeline memory')
args = parser.parse_args()

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_eset_class_labels = robjects.globalenv['esetClassLabels']
r_eset_gene_symbols = robjects.globalenv['esetGeneSymbols']
r_limma_feature_score = robjects.globalenv['limmaFeatureScore']
numpy2ri.activate()
if args.gscv_no_memory:
    memory = None
else:
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)

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
if args.gscv_no_memory:
    limma_score_func = limma
    mi_score_func = mutual_info_classif
    rfe_svm_estimator = LinearSVC(class_weight='balanced')
    sfm_svm_estimator = LinearSVC(penalty='l1', dual=False, class_weight='balanced')
    sfm_ext_estimator = ExtraTreesClassifier(class_weight='balanced')
else:
    limma_score_func = memory.cache(limma)
    mi_score_func = memory.cache(mutual_info_classif)
    rfe_svm_estimator = CachedLinearSVC(class_weight='balanced')
    sfm_svm_estimator = CachedLinearSVC(penalty='l1', dual=False, class_weight='balanced')
    sfm_ext_estimator = CachedExtraTreesClassifier(class_weight='balanced')
gscv_scoring = { 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }
# specify elements in sort order (needed by code dealing with gridsearch cv_results)
if args.clf_svm_c:
    CLF_SVC_C = sorted(args.clf_svm_c)
else:
    CLF_SVC_C = np.logspace(-7, 3, 11)
if args.fs_rfe_c:
    RFE_SVC_C = sorted(args.fs_rfe_c)
else:
    RFE_SVC_C = np.logspace(-7, 2, 10)
if args.fs_sfm_c:
    SFM_SVC_C = sorted(args.fs_sfm_c)
else:
    SFM_SVC_C = np.logspace(-2, 2, 5)
if args.fs_sfm_e:
    SFM_EXT_N_ESTIMATORS = sorted(args.fs_sfm_e)
else:
    SFM_EXT_N_ESTIMATORS = list(range(10, args.fs_sfm_e_max + 1, 10))
if args.fs_sfm_thres:
    SFM_THRESHOLDS = sorted(args.fs_sfm_thres)
else:
    SFM_THRESHOLDS = np.logspace(-9, -5, 5)
if args.fs_skb_k:
    SKB_N_FEATURES = sorted(args.fs_skb_k)
else:
    SKB_N_FEATURES = list(range(1, args.fs_skb_k_max + 1))
if args.fs_rfe_n:
    RFE_N_FEATURES = sorted(args.fs_rfe_n)
else:
    RFE_N_FEATURES = list(range(1, args.fs_rfe_n_max + 1))
if args.fs_fcbf_k:
    FCBF_N_FEATURES = sorted(args.fs_fcbf_k)
else:
    FCBF_N_FEATURES = list(range(1, args.fs_fcbf_k_max + 1))
if args.fs_fpr_p:
    SFP_ALPHA = sorted(args.fs_fpr_p)
else:
    SFP_ALPHA = np.logspace(-3, -2, 2)

pipe = Pipeline([
    ('fs1', None),
    ('slr', None),
    ('fs2', None),
    ('fs3', None),
    ('clf', None),
], memory=memory)
pipelines = {
    'slr': {
        'Standard': {
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
        'SVM-SFM': {
            'steps': [
                ('fs2', SelectFromModel(sfm_svm_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__C': SFM_SVC_C,
                    'fs2__threshold': SFM_THRESHOLDS,
                },
            ],
        },
        'ExtraTrees-SFM': {
            'steps': [
                ('fs2', SelectFromModel(sfm_ext_estimator)),
            ],
            'param_grid': [
                {
                    'fs2__estimator__n_estimators': SFM_EXT_N_ESTIMATORS,
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
                    'fs3__n_features_to_select': RFE_N_FEATURES,
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
                    'fs3__n_features_to_select': RFE_N_FEATURES,
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
                    'fs2__estimator__n_estimators': SFM_EXT_N_ESTIMATORS,
                    'fs2__threshold': SFM_THRESHOLDS,
                    'fs3__estimator__C': RFE_SVC_C,
                    'fs3__n_features_to_select': RFE_N_FEATURES,
                },
            ],
        },
        'SVM-RFE': {
            'steps': [
                ('fs3', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ],
            'param_grid': [
                {
                    'fs3__estimator__C': RFE_SVC_C,
                    'fs3__n_features_to_select': RFE_N_FEATURES,
                },
            ],
        },
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
        'Limma-KBest-ReliefF': {
            'steps': [
                ('fs1', SelectKBest(limma_score_func)),
                ('fs2', ReliefF()),
            ],
            'param_grid': [
                {
                    'fs1__k': SKB_N_FEATURES,
                },
            ],
        },
        'Limma-KBest-CFS': {
            'steps': [
                ('fs1', SelectKBest(limma_score_func)),
                ('fs2', CFS()),
            ],
            'param_grid': [
                {
                    'fs1__k': SKB_N_FEATURES,
                },
            ],
        },
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
    },
}
dataset_names = [
    'gse8894',
    'gse30219',
    'gse31210',
    'gse37745',
    'gse50081'
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
    # 'sva',
    'stica0',
    'stica025',
    'stica05',
    'stica1',
    'svd',
]
fs_methods = [
    'Limma-KBest',
    'MI-KBest',
    'SVM-SFM',
    'ExtraTrees-SFM',
    'Limma-Fpr-SVM-RFE',
    'SVM-SFM-RFE',
    'ExtraTrees-SFM-RFE',
    'Limma-KBest-FCBF',
    # 'SVM-RFE',
    # 'Limma-KBest-ReliefF',
    # 'Limma-KBest-CFS',
]

# analyses
if args.analysis == 1:
    args.datasets_tr = natsorted(args.datasets_tr)
    if args.norm_meth and args.bc_meth:
        dataset_name = '_'.join(args.datasets_tr + [args.norm_meth, args.bc_meth, 'tr'])
    elif args.norm_meth:
        dataset_name = '_'.join(args.datasets_tr + [args.norm_meth, 'tr'])
    elif args.bc_meth:
        dataset_name = '_'.join(args.datasets_tr + [args.bc_meth, 'tr'])
    print('Dataset:', dataset_name)
    eset_name = 'eset_' + dataset_name
    base.load('data/' + eset_name + '.Rda')
    eset = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_name])
    X = np.array(base.t(biobase.exprs(eset)))
    y = np.array(r_eset_class_labels(eset), dtype=int)
    param_grid = []
    for fs_params in pipelines['fs'][args.fs_meth]['param_grid']:
        for slr_params in pipelines['slr'][args.slr_meth]['param_grid']:
            for clf_params in pipelines['clf'][args.clf_meth]['param_grid']:
                params = { **fs_params, **slr_params, **clf_params }
                for (step, object) in \
                    pipelines['fs'][args.fs_meth]['steps'] + \
                    pipelines['slr'][args.slr_meth]['steps'] + \
                    pipelines['clf'][args.clf_meth]['steps'] \
                : params[step] = [ object ]
                param_grid.append(params)
    grid = GridSearchCV(
        pipe, param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
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
        feature_ranks = sorted(
            zip(weights, feature_idxs, feature_names, r_eset_gene_symbols(eset, feature_idxs + 1)),
            reverse=True,
        )
        roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
        bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
        y_score = grid.decision_function(X[te_idxs])
        fpr, tpr, thres = roc_curve(y[te_idxs], y_score, pos_label=1)
        roc_auc_te = roc_auc_score(y[te_idxs], y_score)
        y_pred = grid.predict(X[te_idxs])
        bcr_te = bcr_score(y[te_idxs], y_pred)
        print(
            'Split: %3s' % (split_idx + 1),
            ' ROC AUC (CV/Test): %.4f/%.4f' % (roc_auc_cv, roc_auc_te),
            ' BCR (CV/Test): %.4f/%.4f' % (bcr_cv, bcr_te),
            ' Features: %3s' % feature_idxs.size,
            ' Params:',  {
                k: v for k, v in grid.best_params_.items() if '__' in k
            },
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
                for scorer in gscv_scoring.keys():
                    mean_scores_cv = np.reshape(
                        grid.cv_results_['mean_test_' + scorer][xaxis_group_sorted_idxs], new_shape
                    )
                    std_scores_cv = np.reshape(
                        grid.cv_results_['std_test_' + scorer][xaxis_group_sorted_idxs], new_shape
                    )
                    mean_scores_cv_max_idxs = np.argmax(mean_scores_cv, axis=1)
                    mean_scores_cv = mean_scores_cv[
                        np.arange(len(mean_scores_cv)), mean_scores_cv_max_idxs
                    ]
                    std_scores_cv = std_scores_cv[
                        np.arange(len(std_scores_cv)), mean_scores_cv_max_idxs
                    ]
                    if split_idx == 0:
                        param_cv_scores[param][scorer] = mean_scores_cv
                    else:
                        param_cv_scores[param][scorer] = np.vstack(
                            (param_cv_scores[param][scorer], mean_scores_cv)
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
            lw=4, alpha=0.8, label='Mean ROC AUC'
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
            lw=4, alpha=0.8, label='Mean BCR'
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
        'Mean ROC AUC (CV/Test): %.4f/%.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
        ' Mean BCR (CV/Test): %.4f/%.4f' % (np.mean(roc_aucs_cv), np.mean(roc_aucs_te)),
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
if args.analysis == 2:
    args.datasets_tr = natsorted(args.datasets_tr)
    if args.norm_meth and args.bc_meth:
        dataset_tr_name = '_'.join(args.datasets_tr + [args.norm_meth, args.bc_meth, 'tr'])
    elif args.norm_meth:
        dataset_tr_name = '_'.join(args.datasets_tr + [args.norm_meth, 'tr'])
    elif args.bc_meth:
        dataset_tr_name = '_'.join(args.datasets_tr + [args.bc_meth, 'tr'])
    print('Train Dataset:', dataset_tr_name)
    eset_tr_name = 'eset_' + dataset_tr_name
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
    param_grid = []
    for fs_params in pipelines['fs'][args.fs_meth]['param_grid']:
        for slr_params in pipelines['slr'][args.slr_meth]['param_grid']:
            for clf_params in pipelines['clf'][args.clf_meth]['param_grid']:
                params = { **fs_params, **slr_params, **clf_params }
                for (step, object) in \
                    pipelines['fs'][args.fs_meth]['steps'] + \
                    pipelines['slr'][args.slr_meth]['steps'] + \
                    pipelines['clf'][args.clf_meth]['steps'] \
                : params[step] = [ object ]
                param_grid.append(params)
    grid = GridSearchCV(
        pipe, param_grid=param_grid, scoring=gscv_scoring, refit=args.gscv_refit,
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
        error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X_tr, y_tr)
    dump(grid, 'results/grid_' + dataset_tr_name  + '_' + args.fs_meth.lower() + '.pkl')
    # print summary info
    feature_idxs = np.arange(X_tr.shape[1])
    for step in grid.best_estimator_.named_steps:
        if hasattr(grid.best_estimator_.named_steps[step], 'get_support'):
            feature_idxs = feature_idxs[grid.best_estimator_.named_steps[step].get_support(indices=True)]
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)[feature_idxs]
    if hasattr(grid.best_estimator_.named_steps['clf'], 'coef_'):
        weights = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    elif hasattr(grid.best_estimator_.named_steps['clf'], 'feature_importances_'):
        weights = grid.best_estimator_.named_steps['clf'].feature_importances_
    feature_ranks = sorted(
        zip(weights, feature_idxs, feature_names, r_eset_gene_symbols(eset_tr, feature_idxs + 1)),
        reverse=True,
    )
    roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
    bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
    print(
        'ROC AUC (CV): %.4f' % roc_auc_cv,
        ' BCR (CV): %.4f' % bcr_cv,
        ' Features: %3s' % feature_idxs.size,
        ' Params:',  {
            k: v for k, v in grid.best_params_.items() if '__' in k
        },
    )
    print('Rankings:')
    for weight, _, feature, symbol in feature_ranks: print(feature, '\t', symbol, '\t', weight)
    # pprint(grid.cv_results_, indent=4)
    # pprint(param_grid, indent=4)
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
            mean_roc_aucs_cv = np.reshape(
                grid.cv_results_['mean_test_roc_auc'][xaxis_group_sorted_idxs], new_shape
            )
            std_roc_aucs_cv = np.reshape(
                grid.cv_results_['std_test_roc_auc'][xaxis_group_sorted_idxs], new_shape
            )
            mean_roc_aucs_cv_max_idxs = np.argmax(mean_roc_aucs_cv, axis=1)
            mean_roc_aucs_cv = mean_roc_aucs_cv[
                np.arange(len(mean_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs
            ]
            std_roc_aucs_cv = std_roc_aucs_cv[
                np.arange(len(std_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs
            ]
            mean_bcrs_cv = np.reshape(
                grid.cv_results_['mean_test_bcr'][xaxis_group_sorted_idxs], new_shape
            )
            std_bcrs_cv = np.reshape(
                grid.cv_results_['std_test_bcr'][xaxis_group_sorted_idxs], new_shape
            )
            mean_bcrs_cv_max_idxs = np.argmax(mean_bcrs_cv, axis=1)
            mean_bcrs_cv = mean_bcrs_cv[
                np.arange(len(mean_bcrs_cv)), mean_bcrs_cv_max_idxs
            ]
            std_bcrs_cv = std_bcrs_cv[
                np.arange(len(std_bcrs_cv)), mean_bcrs_cv_max_idxs
            ]
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
            plt.plot(
                x_axis,
                mean_roc_aucs_cv,
                lw=4, alpha=0.8, label='Mean ROC AUC'
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
                lw=4, alpha=0.8, label='Mean BCR'
            )
            plt.fill_between(
                x_axis,
                [m - s for m, s in zip(mean_bcrs_cv, std_bcrs_cv)],
                [m + s for m, s in zip(mean_bcrs_cv, std_bcrs_cv)],
                color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
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
    pipe = Pipeline([
        ('slr', grid.best_params_['slr']),
        ('clf', grid.best_params_['clf']),
    ])
    for dataset_te in natsorted(list(set(dataset_names) - set(args.datasets_tr))):
        if args.no_addon_te:
            dataset_te_name = '_'.join([dataset_te, args.norm_meth])
        else:
            dataset_te_name = '_'.join([dataset_tr_name, dataset_te, 'te'])
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
            y_score = pipe.fit(X_tr[:,top_feature_idxs], y_tr).decision_function(X_te[:,top_feature_idxs])
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc_te = roc_auc_score(y_te, y_score)
            y_pred = pipe.predict(X_te[:,top_feature_idxs])
            bcr_te = bcr_score(y_te, y_pred)
            roc_aucs_te.append(roc_auc_te)
            bcrs_te.append(bcr_te)
        plt.plot(
            x_axis, roc_aucs_te,
            lw=4, alpha=0.8, label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, BCR = %0.4f $\pm$ %0.2f)' % (
                dataset_te_name,
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(bcrs_te), np.std(bcrs_te),
            ),
        )
        # plt.plot(
        #     x_axis, bcrs_te,
        #     lw=4, alpha=0.8,
        # )
        # print summary info
        print(
            'Test Dataset: %3s' % dataset_te_name,
            ' ROC AUC: %.4f' % np.max(roc_aucs_te),
            ' BCR: %.4f' % np.max(bcrs_te),
        )
    plt.legend(loc='lower right', fontsize='small')
    plt.grid('on')
elif args.analysis == 2:
    if args.dataset_tr:
        dataset_pair_names = [t for t in dataset_pair_names if t[0] == args.dataset_tr]
    te_results, bc_results = [], []
    for te_idx, (dataset_tr_name, dataset_te_name) in enumerate(dataset_pair_names):
        for bc_idx, bc_method in enumerate(bc_methods):
            if bc_method != 'none':
                eset_tr_name = 'eset_' + dataset_tr_name + '_tr_' + bc_method
                eset_te_name = eset_tr_name + '_' + dataset_te_name + '_te'
            else:
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
            print(eset_tr_name, '->', eset_te_name)
            base.load('data/' + eset_tr_name + '.Rda')
            eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
            X_tr = np.array(base.t(biobase.exprs(eset_tr)))
            y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
            grid = GridSearchCV(
                Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=memory),
                param_grid=pipelines[args.fs_meth]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if bc_method != 'none':
                dump(grid, 'results/grid_' + dataset_tr_name + '_' + bc_method + '_' + args.fs_meth.lower() + '.pkl')
            else:
                dump(grid, 'results/grid_' + dataset_tr_name + '_' + args.fs_meth.lower() + '.pkl')
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
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
            y_score = grid.decision_function(X_te)
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc_te = roc_auc_score(y_te, y_score)
            y_pred = grid.predict(X_te)
            bcr_te = bcr_score(y_te, y_pred)
            result = {
                'grid': grid,
                'feature_idxs': feature_idxs,
                'feature_names': feature_names,
                'fprs': fpr,
                'tprs': tpr,
                'thres': thres,
                'weights': weights,
                'y_score': y_score,
                'y_test': y_te,
                'roc_auc_cv': roc_auc_cv,
                'roc_auc_te': roc_auc_te,
                'bcr_cv': bcr_cv,
                'bcr_te': bcr_te,
            }
            if te_idx < len(te_results):
                te_results[te_idx].append(result)
            else:
                te_results.append([result])
            if bc_idx < len(bc_results):
                bc_results[bc_idx].append(result)
            else:
                bc_results.append([result])
            base.remove(eset_tr_name)
            base.remove(eset_te_name)
            # print summary info
            print(
                'Features: %3s' % feature_idxs.size,
                ' ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
                ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
                ' Params:',  grid.best_params_,
            )
            print('Rankings:')
            for rank, feature, symbol in sorted(
                zip(weights, feature_names, r_eset_gene_symbols(eset_tr, feature_idxs + 1)),
                reverse=True,
            ): print(feature, '\t', symbol, '\t', rank)
    # plot bc method vs train/test scores
    plt_fig_x_axis = range(1, len(bc_methods) + 1)
    plt.figure('Figure 3-1')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Batch Effect Correction Method on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, bc_methods)
    plt.figure('Figure 3-2')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Batch Effect Correction Method on BCR\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('BCR')
    plt.xticks(plt_fig_x_axis, bc_methods)
    for te_idx, te_bc_results in enumerate(te_results):
        roc_aucs_cv, roc_aucs_te = [], []
        bcrs_cv, bcrs_te, num_features = [], [], []
        for result in te_bc_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            bcrs_cv.append(result['bcr_cv'])
            bcrs_te.append(result['bcr_te'])
            num_features.append(len(result['feature_idxs']))
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure('Figure 3-1')
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                np.mean(roc_aucs_cv), np.std(roc_aucs_cv),
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
        plt.figure('Figure 3-2')
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                np.mean(bcrs_cv), np.std(bcrs_cv),
                np.mean(bcrs_te), np.std(bcrs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
    plt.figure('Figure 3-1')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure('Figure 3-2')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    # plot train/test dataset vs bc method
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure('Figure 4-1')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Train/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure('Figure 4-2')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Train/Held-Out Test Dataset on BCR\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('BCR')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    for bc_idx, bc_te_results in enumerate(bc_results):
        roc_aucs_cv, roc_aucs_te = [], []
        bcrs_cv, bcrs_te, num_features = [], [], []
        for result in bc_te_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            bcrs_cv.append(result['bcr_cv'])
            bcrs_te.append(result['bcr_te'])
            num_features.append(len(result['feature_idxs']))
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure('Figure 4-1')
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                bc_methods[bc_idx],
                np.mean(roc_aucs_cv), np.std(roc_aucs_cv),
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
        plt.figure('Figure 4-2')
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                bc_methods[bc_idx],
                np.mean(bcrs_cv), np.std(bcrs_cv),
                np.mean(bcrs_te), np.std(bcrs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
    plt.figure('Figure 4-1')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure('Figure 4-2')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
elif args.analysis == 3:
    if args.dataset_tr:
        dataset_pair_names = [t for t in dataset_pair_names if t[0] == args.dataset_tr]
    te_results, fs_results = [], []
    for te_idx, (dataset_tr_name, dataset_te_name) in enumerate(dataset_pair_names):
        for fs_idx, fs_method in enumerate(fs_methods):
            if args.bc_meth:
                eset_tr_name = 'eset_' + dataset_tr_name + '_tr_' + args.bc_meth
                eset_te_name = eset_tr_name + '_' + dataset_te_name + '_te'
            else:
                eset_tr_name = 'eset_' + dataset_tr_name
                eset_te_name = 'eset_' + dataset_te_name
            print(eset_tr_name, '->', eset_te_name)
            base.load('data/' + eset_tr_name + '.Rda')
            eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
            X_tr = np.array(base.t(biobase.exprs(eset_tr)))
            y_tr = np.array(r_eset_class_labels(eset_tr), dtype=int)
            grid = GridSearchCV(
                Pipeline(pipelines[fs_method]['pipe_steps'], memory=memory),
                param_grid=pipelines[fs_method]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if args.bc_meth:
                dump(grid, 'results/grid_' + dataset_tr_name + '_' + args.bc_meth + '_' + fs_method.lower() + '.pkl')
            else:
                dump(grid, 'results/grid_' + dataset_tr_name + '_' + fs_method.lower() + '.pkl')
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
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_eset_class_labels(eset_te), dtype=int)
            y_score = grid.decision_function(X_te)
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc_te = roc_auc_score(y_te, y_score)
            y_pred = grid.predict(X_te)
            bcr_te = bcr_score(y_te, y_pred)
            result = {
                'grid': grid,
                'feature_idxs': feature_idxs,
                'feature_names': feature_names,
                'fprs': fpr,
                'tprs': tpr,
                'thres': thres,
                'weights': weights,
                'y_score': y_score,
                'y_test': y_te,
                'roc_auc_cv': roc_auc_cv,
                'roc_auc_te': roc_auc_te,
                'bcr_cv': bcr_cv,
                'bcr_te': bcr_te,
            }
            if te_idx < len(te_results):
                te_results[te_idx].append(result)
            else:
                te_results.append([result])
            if fs_idx < len(fs_results):
                fs_results[fs_idx].append(result)
            else:
                fs_results.append([result])
            base.remove(eset_tr_name)
            base.remove(eset_te_name)
            # print summary info
            print(
                'Features: %3s' % feature_idxs.size,
                ' ROC AUC (CV / Test): %.4f / %.4f' % (roc_auc_cv, roc_auc_te),
                ' BCR (CV / Test): %.4f / %.4f' % (bcr_cv, bcr_te),
                ' Params:',  grid.best_params_,
            )
            print('Rankings:')
            for rank, feature, symbol in sorted(
                zip(weights, feature_names, r_eset_gene_symbols(eset_tr, feature_idxs + 1)),
                reverse=True,
            ): print(feature, '\t', symbol, '\t', rank)
    # plot fs method vs train/test dataset
    plt_fig_x_axis = range(1, len(fs_methods) + 1)
    plt.figure('Figure 5-1')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Feature Selection Method on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Feature Selection Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, fs_methods)
    plt.figure('Figure 5-2')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Feature Selection Method on BCR\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Feature Selection Method')
    plt.ylabel('BCR')
    plt.xticks(plt_fig_x_axis, fs_methods)
    for te_idx, te_fs_results in enumerate(te_results):
        roc_aucs_cv, roc_aucs_te = [], []
        bcrs_cv, bcrs_te, num_features = [], [], []
        for result in te_fs_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            bcrs_cv.append(result['bcr_cv'])
            bcrs_te.append(result['bcr_te'])
            num_features.append(len(result['feature_idxs']))
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure('Figure 5-1')
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                np.mean(roc_aucs_cv), np.std(roc_aucs_cv),
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
        plt.figure('Figure 5-2')
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                np.mean(bcrs_cv), np.std(bcrs_cv),
                np.mean(bcrs_te), np.std(bcrs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
    plt.figure('Figure 5-1')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure('Figure 5-2')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    # plot train/test dataset vs fs method
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure('Figure 6-1')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Training/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure('Figure 6-2')
    plt.rcParams['font.size'] = 16
    plt.title(
        'Effect of Training/Held-Out Test Dataset on BCR\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('BCR')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    for fs_idx, fs_te_results in enumerate(fs_results):
        roc_aucs_cv, roc_aucs_te = [], []
        bcrs_cv, bcrs_te, num_features = [], [], []
        for result in fs_te_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            bcrs_cv.append(result['bcr_cv'])
            bcrs_te.append(result['bcr_te'])
            num_features.append(len(result['feature_idxs']))
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure('Figure 6-1')
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                fs_methods[fs_idx],
                np.mean(roc_aucs_cv), np.std(roc_aucs_cv),
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
        plt.figure('Figure 6-2')
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                fs_methods[fs_idx],
                np.mean(bcrs_cv), np.std(bcrs_cv),
                np.mean(bcrs_te), np.std(bcrs_te),
                np.mean(num_features), np.std(num_features),
            )
        )
    plt.figure('Figure 6-1')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure('Figure 6-2')
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')

plt.show()
if not args.gscv_no_memory: rmtree(cachedir)
