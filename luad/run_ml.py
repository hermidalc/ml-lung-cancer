#!/usr/bin/env python

import argparse, pprint
from tempfile import mkdtemp
from shutil import rmtree
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
# from rpy2.robjects import pandas2ri
# import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFpr, SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import dump, Memory
from feature_selection import CFS
import matplotlib.pyplot as plt
from matplotlib import style

# config
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--bc-meth', type=str, help='batch effect correction method')
parser.add_argument('--fs-meth', type=str, help='feature selection method')
parser.add_argument('--fs-num-max', type=int, default=30, help='fs num max')
parser.add_argument('--fs-num-select', type=int, nargs="+", help='fs num select')
parser.add_argument('--fs-fpr-pval', type=float, nargs="+", help='fs fpr p-value')
parser.add_argument('--fs-sfm-thres', type=float, nargs="+", help='fs sfm threshold')
parser.add_argument('--fs-sfm-c', type=float, nargs="+", help='fs sfm c')
parser.add_argument('--fs-rfe-step', type=float, default=0.2, help='fs rfe step')
parser.add_argument('--fs-rfe-verbose', type=int, default=0, help='fs rfe verbosity')
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help='fs rank method (mean_coefs or mean_roc_aucs)')
parser.add_argument('--clf-svm-c', type=float, nargs="+", help='clf svm c')
parser.add_argument('--gscv-splits', type=int, default=30, help='gscv splits')
parser.add_argument('--gscv-size', type=int, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--gscv-refit', type=str, default='roc_auc', help='gscv refit score function (roc_auc or bcr)')
parser.add_argument('--dataset-tr', type=str, help='dataset fs/tr')
args = parser.parse_args()

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_filter_eset_relapse_labels = robjects.globalenv['filterEsetRelapseLabels']
r_get_gene_symbols = robjects.globalenv['getGeneSymbols']
r_limma = robjects.globalenv['limma']
numpy2ri.activate()
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

# custom mixin and class for caching pipeline nested LinearSVC fits
class CachedFitMixin:
    def fit(self, *args, **kwargs):
        fit = memory.cache(super(CachedFitMixin, self).fit)
        cached_self = fit(*args, **kwargs)
        vars(self).update(vars(cached_self))
        return self

class CachedLinearSVC(CachedFitMixin, LinearSVC):
    pass

# limma feature selection scoring function
def limma(X, y):
    f, pv = r_limma(np.transpose(X), y)
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
limma_cached = memory.cache(limma)
mutual_info_classif_cached = memory.cache(mutual_info_classif)

# specify elements in sort order (needed by code dealing with gridsearch cv_results)
if args.clf_svm_c:
    CLF_SVC_C = sorted(args.clf_svm_c)
else:
    CLF_SVC_C = [ 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 ]
if args.fs_sfm_c:
    SFM_SVC_C = sorted(args.fs_sfm_c)
else:
    SFM_SVC_C = [ 1e-2, 1e-1, 1, 10, 10 ]
if args.fs_sfm_thres:
    SFM_THRESHOLDS = sorted(args.fs_sfm_thres)
else:
    SFM_THRESHOLDS = [ 0.01, 0.02, 0.03, 0.04 ]
if args.fs_num_select:
    SKB_N_FEATURES = sorted(args.fs_num_select)
else:
    SKB_N_FEATURES = list(range(1, args.fs_num_max + 1))
if args.fs_num_select:
    RFE_N_FEATURES = sorted(args.fs_num_select)
else:
    RFE_N_FEATURES = list(range(5, args.fs_num_max + 1, 5))
if args.fs_fpr_pval:
    SFP_ALPHA = sorted(args.fs_fpr_pval)
else:
    SFP_ALPHA = [ 0.001, 0.01 ]

pipelines = {
    'Limma-KBest': {
        'pipe_steps': [
            ('fsl', SelectKBest(limma_cached)),
            ('slr', StandardScaler()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__k': SKB_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'MI-KBest': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('fsl', SelectKBest(mutual_info_classif_cached)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__k': SKB_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-Fpr-SVM-RFE': {
        'pipe_steps': [
            ('sfp', SelectFpr(limma_cached)),
            ('slr', StandardScaler()),
            ('fsl', RFE(
                CachedLinearSVC(class_weight='balanced'),
                step=args.fs_rfe_step, verbose=args.fs_rfe_verbose,
            )),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfp__alpha': SFP_ALPHA,
                'fsl__n_features_to_select': RFE_N_FEATURES,
                'fsl__estimator__C': CLF_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-SFM-RFE': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('sfm', SelectFromModel(
                CachedLinearSVC(penalty='l1', dual=False, class_weight='balanced'),
            )),
            ('fsl', RFE(
                CachedLinearSVC(class_weight='balanced'),
                step=args.fs_rfe_step, verbose=args.fs_rfe_verbose,
            )),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfm__threshold': SFM_THRESHOLDS,
                'sfm__estimator__C': SFM_SVC_C,
                'fsl__n_features_to_select': RFE_N_FEATURES,
                'fsl__estimator__C': CLF_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-RFE': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('fsl', RFE(
                CachedLinearSVC(class_weight='balanced'),
                step=args.fs_rfe_step, verbose=args.fs_rfe_verbose,
            )),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__n_features_to_select': RFE_N_FEATURES,
                'fsl__estimator__C': CLF_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-SFM': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('fsl', SelectFromModel(
                CachedLinearSVC(penalty='l1', dual=False, class_weight='balanced')
            )),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__threshold': SFM_THRESHOLDS,
                'fsl__estimator__C': SFM_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'ExtraTrees-SFM': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('fsl', SelectFromModel(ExtraTreesClassifier())),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__threshold': SFM_THRESHOLDS,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-Fpr-CFS': {
        'pipe_steps': [
            ('sfp', SelectFpr(limma_cached)),
            ('slr', StandardScaler()),
            ('fsl', CFS()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfp__alpha': SFP_ALPHA,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
}
dataset_pair_names = [
    # ('gse31210_gse30219', 'gse8894'),
    # ('gse31210_gse8894', 'gse30219'),
    # ('gse8894_gse30219', 'gse31210'),
    ('gse31210_gse30219_gse37745', 'gse8894'),
    ('gse31210_gse8894_gse37745', 'gse30219'),
    ('gse8894_gse30219_gse37745', 'gse31210'),
    ('gse31210_gse8894_gse30219', 'gse37745'),
    # ('gse31210_gse8894_gse30219_gse37745', 'gse50081'),
    # ('gse31210_gse8894_gse30219_gse50081', 'gse37745'),
    # ('gse31210_gse8894_gse37745_gse50081', 'gse30219'),
    # ('gse31210_gse30219_gse37745_gse50081', 'gse8894'),
    # ('gse8894_gse30219_gse37745_gse50081', 'gse31210'),
]
bc_methods = [
    'none',
    'std',
    'cbt',
    #'fab',
    'sva',
    'stica0',
    'stica025',
    'stica05',
    'stica1',
    'svd',
]
fs_methods = [
    'Limma-KBest',
    'MI-KBest',
    'Limma-Fpr-SVM-RFE',
    #'SVM-SFM-RFE',
    #'SVM-RFE',
    #'SVM-SFM',
    #'ExtraTrees-SFM',
    'Limma-Fpr-CFS',
]
gscv_scoring = { 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }

# analyses
if args.analysis == 1:
    if args.bc_meth:
        eset_tr_name = 'eset_' + args.dataset_tr + '_tr_' + args.bc_meth
    else:
        eset_tr_name = 'eset_' + args.dataset_tr
    print(eset_tr_name)
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=memory),
        param_grid=pipelines[args.fs_meth]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
        error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X_tr, y_tr)
    if args.bc_meth:
        dump(grid, 'data/grid_' + args.dataset_tr + '_' + args.bc_meth + '_' + args.fs_meth.lower() + '.pkl')
    else:
        dump(grid, 'data/grid_' + args.dataset_tr + '_' + args.fs_meth.lower() + '.pkl')
    # print summary info
    feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
    feature_names = feature_names[feature_idxs]
    coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    feature_ranks = sorted(
        zip(
            coefs, feature_idxs, feature_names,
            r_get_gene_symbols(
                eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
            ),
        ),
        reverse=True
    )
    roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
    bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
    print(
        'Features: %3s' % feature_idxs.size,
        ' ROC AUC (CV): %.4f' % roc_auc_cv,
        ' BCR (CV): %.4f' % bcr_cv,
        ' Params:',  grid.best_params_,
    )
    print('Rankings:')
    for coef, _, feature, symbol in feature_ranks: print(feature, '\t', symbol, '\t', coef)
    grid_params = pipelines[args.fs_meth]['param_grid'][0]
    # plot num top-ranked features selected vs cv perf metrics
    if args.fs_meth in ('Limma-KBest', 'MI-KBest'):
        new_shape = (
            len(grid_params['fsl__k']),
            np.prod([len(v) for k,v in grid_params.items() if k != 'fsl__k'])
        )
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__k'])
        )
    elif args.fs_meth in ('Limma-Fpr-SVM-RFE', 'SVM-RFE'):
        new_shape = (
            len(grid_params['fsl__n_features_to_select']),
            np.prod([len(v) for k,v in grid_params.items() if k != 'fsl__n_features_to_select'])
        )
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__n_features_to_select'])
        )
    elif args.fs_meth in ('SVM-SFM', 'ExtraTrees-SFM'):
        new_shape = (
            len(grid_params['fsl__threshold']),
            np.prod([len(v) for k,v in grid_params.items() if k != 'fsl__threshold'])
        )
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__threshold']).astype(str)
        )
    elif args.fs_meth in ('Limma-Fpr-CFS'):
        new_shape = (
            len(grid_params['sfp__alpha']),
            np.prod([len(v) for k,v in grid_params.items() if k != 'sfp__alpha'])
        )
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_sfp__alpha'])
        )
    mean_roc_aucs_cv = np.reshape(grid.cv_results_['mean_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    std_roc_aucs_cv = np.reshape(grid.cv_results_['std_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    mean_roc_aucs_cv_max_idxs = np.argmax(mean_roc_aucs_cv, axis=1)
    mean_roc_aucs_cv = mean_roc_aucs_cv[np.arange(len(mean_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs]
    std_roc_aucs_cv = std_roc_aucs_cv[np.arange(len(std_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs]
    mean_bcrs_cv = np.reshape(grid.cv_results_['mean_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    std_bcrs_cv = np.reshape(grid.cv_results_['std_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    mean_bcrs_cv_max_idxs = np.argmax(mean_bcrs_cv, axis=1)
    mean_bcrs_cv = mean_bcrs_cv[np.arange(len(mean_bcrs_cv)), mean_bcrs_cv_max_idxs]
    std_bcrs_cv = std_bcrs_cv[np.arange(len(std_bcrs_cv)), mean_bcrs_cv_max_idxs]
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    dataset_tr_name = args.dataset_tr.replace('gse', 'GSE')
    if args.bc_meth:
        dataset_tr_name = dataset_tr_name + ' ' + args.bc_meth
    plt.title(
        dataset_tr_name + ' SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected on CV Performance Metrics'
    )
    plt.xlabel('Number of Top-Ranked Features Selected')
    plt.ylabel('CV Score')
    if args.fs_meth in ('Limma-KBest', 'MI-KBest'):
        x_axis = grid_params['fsl__k']
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth in ('Limma-Fpr-SVM-RFE', 'SVM-RFE'):
        x_axis = grid_params['fsl__n_features_to_select']
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth in ('SVM-SFM', 'ExtraTrees-SFM'):
        x_axis = range(len(grid_params['fsl__threshold']))
        plt.xticks(x_axis, grid_params['fsl__threshold'])
    elif args.fs_meth in ('Limma-Fpr-CFS'):
        x_axis = grange(len(grid_params['sfp__alpha']))
        plt.xticks(x_axis, grid_params['sfp__alpha'])
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
    # plot svm c vs cv perf metrics
    new_shape = (
        len(grid_params['clf__C']),
        np.prod([len(v) for k,v in grid_params.items() if k != 'clf__C'])
    )
    xaxis_group_sorted_idxs = np.argsort(
        np.ma.getdata(grid.cv_results_['param_clf__C'])
    )
    mean_roc_aucs_cv = np.reshape(grid.cv_results_['mean_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    std_roc_aucs_cv = np.reshape(grid.cv_results_['std_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    mean_roc_aucs_cv_max_idxs = np.argmax(mean_roc_aucs_cv, axis=1)
    mean_roc_aucs_cv = mean_roc_aucs_cv[np.arange(len(mean_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs]
    std_roc_aucs_cv = std_roc_aucs_cv[np.arange(len(std_roc_aucs_cv)), mean_roc_aucs_cv_max_idxs]
    mean_bcrs_cv = np.reshape(grid.cv_results_['mean_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    std_bcrs_cv = np.reshape(grid.cv_results_['std_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    mean_bcrs_cv_max_idxs = np.argmax(mean_bcrs_cv, axis=1)
    mean_bcrs_cv = mean_bcrs_cv[np.arange(len(mean_bcrs_cv)), mean_bcrs_cv_max_idxs]
    std_bcrs_cv = std_bcrs_cv[np.arange(len(std_bcrs_cv)), mean_bcrs_cv_max_idxs]
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    dataset_tr_name = args.dataset_tr.replace('gse', 'GSE')
    if args.bc_meth:
        dataset_tr_name = dataset_tr_name + ' ' + args.bc_meth
    plt.title(
        dataset_tr_name + ' SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of SVM C Hyperparameter on CV Performance Metrics'
    )
    plt.xlabel('SVM C')
    plt.ylabel('CV Score')
    x_axis = range(len(grid_params['clf__C']))
    plt.xticks(x_axis, grid_params['clf__C'])
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
    plt.figure(3)
    plt.rcParams['font.size'] = 20
    dataset_tr_name = args.dataset_tr.replace('gse', 'GSE')
    if args.bc_meth:
        dataset_tr_name = dataset_tr_name + ' ' + args.bc_meth
    plt.title(
        dataset_tr_name + ' SVM Classifier (' + args.fs_meth + 'Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected Performance Metrics'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('Test Score')
    x_axis = range(1, feature_idxs.size + 1)
    plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
    plt.xticks(x_axis)
    dataset_te_names = []
    for dataset_tr_name, dataset_te_name in dataset_pair_names:
        if args.dataset_tr == dataset_tr_name:
            dataset_te_names = [dataset_te_name]
            break
        elif args.dataset_tr != dataset_te_name:
            dataset_te_names.append(dataset_te_name)
    ranked_feature_idxs = [x for _, x, _, _ in feature_ranks]
    clf = Pipeline([
        ('slr', StandardScaler()),
        ('clf', LinearSVC(
            class_weight='balanced', C=grid.best_params_['clf__C'],
        )),
    ])
    for dataset_te_name in dataset_te_names:
        if args.bc_meth:
            eset_te_name = eset_tr_name + '_' + dataset_te_name + '_te'
        else:
            eset_te_name = 'eset_' + dataset_te_name
        base.load('data/' + eset_te_name + '.Rda')
        eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        X_te = np.array(base.t(biobase.exprs(eset_te)))
        y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
        roc_aucs_te, bcrs_te = [], []
        for num_features in range(1, len(ranked_feature_idxs) + 1):
            top_feature_idxs = ranked_feature_idxs[:num_features]
            top_feature_names = ranked_feature_idxs[:num_features]
            y_score = clf.fit(X_tr[:,top_feature_idxs], y_tr).decision_function(X_te[:,top_feature_idxs])
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc_te = roc_auc_score(y_te, y_score)
            y_pred = clf.predict(X_te[:,top_feature_idxs])
            bcr_te = bcr_score(y_te, y_pred)
            roc_aucs_te.append(roc_auc_te)
            bcrs_te.append(bcr_te)
        plt.plot(
            x_axis, roc_aucs_te,
            lw=4, alpha=0.8, label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, BCR = %0.4f $\pm$ %0.2f)' % (
                dataset_te_name.replace('gse', 'GSE'),
                np.mean(roc_aucs_te), np.std(roc_aucs_te),
                np.mean(bcrs_te), np.std(bcrs_te),
            ),
        )
        plt.plot(
            x_axis, bcrs_te,
            lw=4, alpha=0.8,
        )
        # print summary info
        print(
            'Dataset: %3s' % eset_te_name,
            ' ROC AUC: %.4f' % np.max(roc_aucs_te),
            ' BCR: %.4f' % np.max(bcrs_te),
        )
    plt.legend(loc='lower right', fontsize='small')
    plt.grid('on')
elif args.analysis == 2:
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
            y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
            grid = GridSearchCV(
                Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=memory),
                param_grid=pipelines[args.fs_meth]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if bc_method != 'none':
                dump(grid, 'data/grid_' + dataset_tr_name + '_' + bc_method + '_' + args.fs_meth.lower() + '.pkl')
            else:
                dump(grid, 'data/grid_' + dataset_tr_name + '_' + args.fs_meth.lower() + '.pkl')
            feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
            feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
            feature_names = feature_names[feature_idxs]
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
            bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
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
                'coefs': coefs,
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
                zip(
                    coefs,
                    feature_names,
                    r_get_gene_symbols(
                        eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                    ),
                ),
                reverse=True
            ): print(feature, '\t', symbol, '\t', rank)
    # plot bc method vs train/test scores
    plt_fig_x_axis = range(1, len(bc_methods) + 1)
    plt.figure(4)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, bc_methods)
    plt.figure(5)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on BCR\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
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
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_bcr_cv = np.mean(bcrs_cv)
        mean_bcr_te = np.mean(bcrs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_bcr_cv = np.std(bcrs_cv)
        std_bcr_te = np.std(bcrs_te)
        std_num_features = np.std(num_features)
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure(4)
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_roc_auc_cv, std_roc_auc_cv,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
        plt.figure(5)
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_bcr_cv, std_bcr_cv,
                mean_bcr_te, std_bcr_te,
                mean_num_features, std_num_features,
            )
        )
    plt.figure(4)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure(5)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    # plot train/test dataset vs bc method
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure(6)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Train/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure(7)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Train/Held-Out Test Dataset on BCR\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
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
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_bcr_cv = np.mean(bcrs_cv)
        mean_bcr_te = np.mean(bcrs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_bcr_cv = np.std(bcrs_cv)
        std_bcr_te = np.std(bcrs_te)
        std_num_features = np.std(num_features)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure(6)
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                bc_methods[bc_idx],
                mean_roc_auc_cv, std_roc_auc_cv,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
        plt.figure(7)
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                bc_methods[bc_idx],
                mean_bcr_cv, std_bcr_cv,
                mean_bcr_te, std_bcr_te,
                mean_num_features, std_num_features,
            )
        )
    plt.figure(6)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure(7)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
elif args.analysis == 3:
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
            y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
            grid = GridSearchCV(
                Pipeline(pipelines[fs_method]['pipe_steps'], memory=memory),
                param_grid=pipelines[fs_method]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size), iid=False,
                error_score=0, return_train_score=False, n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if args.bc_meth:
                dump(grid, 'data/grid_' + dataset_tr_name + '_' + args.bc_meth + '_' + fs_method.lower() + '.pkl')
            else:
                dump(grid, 'data/grid_' + dataset_tr_name + '_' + fs_method.lower() + '.pkl')
            feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
            feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
            feature_names = feature_names[feature_idxs]
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
            bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
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
                'coefs': coefs,
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
                zip(
                    coefs,
                    feature_names,
                    r_get_gene_symbols(
                        eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                    ),
                ),
                reverse=True
            ): print(feature, '\t', symbol, '\t', rank)
    # plot fs method vs train/test dataset
    plt_fig_x_axis = range(1, len(fs_methods) + 1)
    plt.figure(8)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Feature Selection Method on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
    )
    plt.xlabel('Feature Selection Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, fs_methods)
    plt.figure(9)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Feature Selection Method on BCR\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
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
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_bcr_cv = np.mean(bcrs_cv)
        mean_bcr_te = np.mean(bcrs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_bcr_cv = np.std(bcrs_cv)
        std_bcr_te = np.std(bcrs_te)
        std_num_features = np.std(num_features)
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure(8)
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_roc_auc_cv, std_roc_auc_cv,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
        plt.figure(9)
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_bcr_cv, std_bcr_cv,
                mean_bcr_te, std_bcr_te,
                mean_num_features, std_num_features,
            )
        )
    plt.figure(8)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure(9)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    # plot train/test dataset vs fs method
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure(10)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure(11)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on BCR\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
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
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_bcr_cv = np.mean(bcrs_cv)
        mean_bcr_te = np.mean(bcrs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_bcr_cv = np.std(bcrs_cv)
        std_bcr_te = np.std(bcrs_te)
        std_num_features = np.std(num_features)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.figure(10)
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                fs_methods[bc_idx],
                mean_roc_auc_cv, std_roc_auc_cv,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
        plt.figure(11)
        plt.plot(
            plt_fig_x_axis, bcrs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
        )
        plt.plot(
            plt_fig_x_axis, bcrs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (CV = %0.4f $\pm$ %0.2f, Test = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                fs_methods[bc_idx],
                mean_bcr_cv, std_bcr_cv,
                mean_bcr_te, std_bcr_te,
                mean_num_features, std_num_features,
            )
        )
    plt.figure(10)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')
    plt.figure(11)
    plt.legend(loc='best', fontsize='x-small')
    plt.grid('on')

plt.show()
rmtree(cachedir)
