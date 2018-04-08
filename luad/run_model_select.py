#!/usr/bin/env python

import argparse, pprint
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
from feature_selection import CFS, FCBF
import matplotlib.pyplot as plt
from matplotlib import style

# config
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--norm-meth', type=str, help='preprocess/normalization method')
parser.add_argument('--bc-meth', type=str, help='batch effect correction method')
parser.add_argument('--fs-meth', type=str, help='feature selection method')
parser.add_argument('--fs-skb-k', type=int, nargs='+', help='fs skb k select')
parser.add_argument('--fs-skb-k-max', type=int, default=30, help='fs skb k max')
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
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help='fs rank method (mean_coefs or mean_roc_aucs)')
parser.add_argument('--clf-svm-c', type=float, nargs='+', help='clf svm c')
parser.add_argument('--gscv-splits', type=int, default=80, help='gscv splits')
parser.add_argument('--gscv-size', type=float, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--gscv-refit', type=str, default='roc_auc', help='gscv refit score function (roc_auc or bcr)')
parser.add_argument('--gscv-no-memory', default=False, action='store_true', help='gscv no pipeline memory')
parser.add_argument('--dataset-tr-num', type=int, help='dataset tr num combos')
parser.add_argument('--datasets-tr', type=str, nargs='+', help='datasets tr')
parser.add_argument('--datasets-te', type=str, nargs='+', help='datasets te')
args = parser.parse_args()

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_get_eset_class_labels = robjects.globalenv['getEsetClassLabels']
r_get_eset_gene_symbols = robjects.globalenv['getEsetGeneSymbols']
r_limma_feature_score = robjects.globalenv['limmaFeatureScore']
numpy2ri.activate()
if args.gscv_no_memory:
    memory = None
else:
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
if args.fs_fpr_p:
    SFP_ALPHA = sorted(args.fs_fpr_p)
else:
    SFP_ALPHA = np.logspace(-3, -2, 2)

pipelines = {
    'Limma-KBest': {
        'pipe_steps': [
            ('skb', SelectKBest(limma_score_func)),
            ('std', StandardScaler()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'skb__k': SKB_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'MI-KBest': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('skb', SelectKBest(mi_score_func)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'skb__k': SKB_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-Fpr-SVM-RFE': {
        'pipe_steps': [
            ('sfp', SelectFpr(limma_score_func)),
            ('std', StandardScaler()),
            ('rfe', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfp__alpha': SFP_ALPHA,
                'rfe__estimator__C': RFE_SVC_C,
                'rfe__n_features_to_select': RFE_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-SFM-RFE': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('sfm', SelectFromModel(sfm_svm_estimator)),
            ('rfe', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfm__estimator__C': SFM_SVC_C,
                'sfm__threshold': SFM_THRESHOLDS,
                'rfe__n_features_to_select': RFE_N_FEATURES,
                'rfe__estimator__C': RFE_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'ExtraTrees-SFM-RFE': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('sfm', SelectFromModel(sfm_ext_estimator)),
            ('rfe', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfm__estimator__n_estimators': SFM_EXT_N_ESTIMATORS,
                'sfm__threshold': SFM_THRESHOLDS,
                'rfe__n_features_to_select': RFE_N_FEATURES,
                'rfe__estimator__C': RFE_SVC_C,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-RFE': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('rfe', RFE(rfe_svm_estimator, step=args.fs_rfe_step, verbose=args.fs_rfe_verbose)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'rfe__estimator__C': RFE_SVC_C,
                'rfe__n_features_to_select': RFE_N_FEATURES,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'SVM-SFM': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('sfm', SelectFromModel(sfm_svm_estimator)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfm__estimator__C': SFM_SVC_C,
                'sfm__threshold': SFM_THRESHOLDS,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'ExtraTrees-SFM': {
        'pipe_steps': [
            ('std', StandardScaler()),
            ('sfm', SelectFromModel(sfm_ext_estimator)),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfm__estimator__n_estimators': SFM_EXT_N_ESTIMATORS,
                'sfm__threshold': SFM_THRESHOLDS,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-Fpr-CFS': {
        'pipe_steps': [
            ('sfp', SelectFpr(limma_score_func)),
            ('std', StandardScaler()),
            ('cfs', CFS()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'sfp__alpha': SFP_ALPHA,
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-KBest-CFS': {
        'pipe_steps': [
            ('skb', SelectKBest(limma_score_func)),
            ('std', StandardScaler()),
            ('cfs', CFS()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'skb__k': [ 200 ],
                'clf__C': CLF_SVC_C,
            },
        ],
    },
    'Limma-KBest-FCBF': {
        'pipe_steps': [
            ('skb', SelectKBest(limma_score_func)),
            ('std', StandardScaler()),
            ('fcbf', FCBF()),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'skb__k': [ 10000 ],
                'clf__C': CLF_SVC_C,
            },
        ],
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
    #'fab',
    #'sva',
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
    'SVM-SFM-RFE',
    'ExtraTrees-SFM-RFE',
    'Limma-KBest-FCBF',
    #'SVM-RFE',
    #'SVM-SFM',
    #'ExtraTrees-SFM',
    #'Limma-Fpr-CFS',
    #'Limma-KBest-CFS',
]

# analyses
if args.analysis == 1:
    args.datasets_tr = sorted(args.datasets_tr)
    if args.norm_meth and args.bc_meth:
        dataset_tr_name = '_'.join(args.datasets_tr.extend(args.norm_meth, args.bc_meth, 'tr'))
    elif args.norm_meth:
        dataset_tr_name = '_'.join(args.datasets_tr.extend(args.norm_meth, 'tr'))
    elif args.bc_meth:
        dataset_tr_name = '_'.join(args.datasets_tr.extend(args.bc_meth, 'tr'))
    eset_tr_name = 'eset_' + dataset_tr_name
    print(eset_tr_name)
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_get_eset_class_labels(eset_tr), dtype=int)
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=memory),
        param_grid=pipelines[args.fs_meth]['param_grid'], scoring=gscv_scoring, refit=args.gscv_refit,
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
    coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    feature_ranks = sorted(
        zip(
            coefs, feature_idxs, feature_names,
            r_get_eset_gene_symbols(
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
    # plot grid search parameters vs cv perf metrics
    dataset_tr_title = dataset_tr_name.replace('gse', 'GSE')
    grid_params = pipelines[args.fs_meth]['param_grid'][0]
    for idx, param in enumerate(grid_params):
        new_shape = ()
        if param in (
            'skb__k', 'rfe__n_features_to_select', 'sfm__estimator__C', 'sfm__estimator__n_estimators',
            'sfm__threshold', 'sfp__alpha', 'rfe__estimator__C', 'clf__C',
        ) and len(grid_params[param]) > 1:
            new_shape = (
                len(grid_params[param]),
                np.prod([len(v) for k,v in grid_params.items() if k != param])
            )
            if param in ('sfm__threshold'):
                xaxis_group_sorted_idxs = np.argsort(
                    np.ma.getdata(grid.cv_results_['param_' + param]).astype(str)
                )
            else:
                xaxis_group_sorted_idxs = np.argsort(
                    np.ma.getdata(grid.cv_results_['param_' + param])
                )
        if new_shape:
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
            plt.figure('Figure 1-' + str(idx + 1))
            plt.rcParams['font.size'] = 20
            if param in (
                'skb__k', 'rfe__n_features_to_select', 'sfm__estimator__n_estimators',
            ):
                x_axis = grid_params[param]
                plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
                plt.xticks(x_axis)
            elif param in (
                'sfm__estimator__C', 'sfm__threshold', 'sfp__alpha', 'rfe__estimator__C', 'clf__C',
            ):
                x_axis = range(len(grid_params[param]))
                plt.xticks(x_axis, grid_params[param])
            plt.title(
                dataset_tr_title + ' SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
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
    plt.figure('Figure 2')
    plt.rcParams['font.size'] = 20
    plt.title(
        dataset_tr_title + ' SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected Performance Metrics'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('Test Score')
    x_axis = range(1, feature_idxs.size + 1)
    plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
    plt.xticks(x_axis)
    ranked_feature_idxs = [x for _, x, _, _ in feature_ranks]
    clf = Pipeline([
        ('std', StandardScaler()),
        ('clf', LinearSVC(
            class_weight='balanced', C=grid.best_params_['clf__C'],
        )),
    ])
    for dataset_te_name in sorted(list(set(dataset_names) - set(args.datasets_tr))):
        if args.norm_meth or args.bc_meth:
            eset_te_name = eset_tr_name + '_' + dataset_te_name + '_te'
        else:
            eset_te_name = 'eset_' + dataset_te_name
        eset_te_file = 'data/' + eset_te_name + '.Rda'
        if not path.isfile(eset_te_file): continue
        base.load(eset_te_file)
        eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        X_te = np.array(base.t(biobase.exprs(eset_te)))
        y_te = np.array(r_get_eset_class_labels(eset_te), dtype=int)
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
            'Dataset: %3s' % dataset_te_name,
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
            y_tr = np.array(r_get_eset_class_labels(eset_tr), dtype=int)
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
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
            bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_get_eset_class_labels(eset_te), dtype=int)
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
                    r_get_eset_gene_symbols(
                        eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                    ),
                ),
                reverse=True
            ): print(feature, '\t', symbol, '\t', rank)
    # plot bc method vs train/test scores
    plt_fig_x_axis = range(1, len(bc_methods) + 1)
    plt.figure('Figure 3-1')
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, bc_methods)
    plt.figure('Figure 3-2')
    plt.rcParams['font.size'] = 20
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
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
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
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure('Figure 4-1')
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Train/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.fs_meth + ' Feature Selection)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure('Figure 4-2')
    plt.rcParams['font.size'] = 20
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
            y_tr = np.array(r_get_eset_class_labels(eset_tr), dtype=int)
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
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            roc_auc_cv = grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
            bcr_cv = grid.cv_results_['mean_test_bcr'][grid.best_index_]
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_get_eset_class_labels(eset_te), dtype=int)
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
                    r_get_eset_gene_symbols(
                        eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                    ),
                ),
                reverse=True
            ): print(feature, '\t', symbol, '\t', rank)
    # plot fs method vs train/test dataset
    plt_fig_x_axis = range(1, len(fs_methods) + 1)
    plt.figure('Figure 5-1')
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Feature Selection Method on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Feature Selection Method')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, fs_methods)
    plt.figure('Figure 5-2')
    plt.rcParams['font.size'] = 20
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
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
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
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.figure('Figure 6-1')
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on ROC AUC\n' +
        '(' + args.bc_meth + ' Batch Effect Correction)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    plt.figure('Figure 6-2')
    plt.rcParams['font.size'] = 20
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
