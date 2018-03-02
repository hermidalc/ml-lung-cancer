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
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals import joblib
from feature_selection import CFS
import matplotlib.pyplot as plt
from matplotlib import style

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_filter_eset_relapse_labels = robjects.globalenv['filterEsetRelapseLabels']
r_get_gene_symbols = robjects.globalenv['getGeneSymbols']
r_limma = robjects.globalenv['limma']
cachedir = mkdtemp()
numpy2ri.activate()

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
    # if presence of only one class
    if mes2 == 0:
        return tp / mes1
    elif mes1 == 0:
        return tn / mes2
    else:
        return (tp / mes1 + tn / mes2) / 2

# config
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--fs-meth', type=str, help='feature selection method')
parser.add_argument('--fs-num-max', type=int, default=30, help='fs num max')
parser.add_argument('--fs-num-final', type=int, default=20, help='fs num final')
parser.add_argument('--fs-fpr-pval', type=float, default=0.01, help='fs fpr min p-value')
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help='mean_coefs or mean_roc_aucs')
parser.add_argument('--gscv-splits', type=int, default=30, help='gscv splits')
parser.add_argument('--gscv-size', type=int, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--rfe-step', type=float, default=0.01, help='rfe step')
parser.add_argument('--rfe-verbose', type=int, default=0, help='rfe verbosity')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help='svm algorithm (liblinear or libsvm)')
parser.add_argument('--dataset-tr', type=str, help='dataset fs/tr')
parser.add_argument('--dataset-te', type=str, nargs="+", help='dataset te')
parser.add_argument('--bc-meth', type=str, help='batch effect correction method')
args = parser.parse_args()

dataset_pair_names = [
    ('gse31210_gse30219', 'gse8894'),
    ('gse31210_gse8894', 'gse30219'),
    ('gse8894_gse30219', 'gse31210'),
    # ('gse31210_gse30219_gse37745', 'gse8894'),
    # ('gse31210_gse8894_gse37745', 'gse30219'),
    # ('gse8894_gse30219_gse37745', 'gse31210'),
    # ('gse31210_gse8894_gse30219', 'gse37745'),
]

# specify in sort order (needed by code dealing with gridsearch cv_results)
CLF_SVC_C = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
SFM_SVC_C = [ 0.01, 0.1, 1, 10, 100, 1000 ]
SFM_THRESHOLDS = [ 0.01, 0.02, 0.03, 0.04 ]
SKB_N_FEATURES = list(range(1, args.fs_num_max + 1))
RFE_N_FEATURES = list(range(5, args.fs_num_max + 1, 5))
FPR_ALPHA = [ 0.001, 0.01, 0.05 ]

pipelines = {
    'Limma-KBest': {
        'pipe_steps': [
            ('fsl', SelectKBest(limma)),
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
            ('fsl', SelectKBest(mutual_info_classif)),
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
            ('sfp', SelectFpr(limma, alpha=args.fs_fpr_pval)),
            ('slr', StandardScaler()),
            ('fsl', RFE(
                LinearSVC(class_weight='balanced'),
                step=args.rfe_step, verbose=args.rfe_verbose,
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
    'SVM-RFE': {
        'pipe_steps': [
            ('slr', StandardScaler()),
            ('fsl', RFE(
                LinearSVC(class_weight='balanced'),
                step=args.rfe_step, verbose=args.rfe_verbose,
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
                LinearSVC(penalty='l1', dual=False, class_weight='balanced')
            )),
            ('clf', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'fsl__estimator__C': SFM_SVC_C,
                'fsl__threshold': SFM_THRESHOLDS,
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
            ('sfp', SelectFpr(limma, alpha=args.fs_fpr_pval)),
            ('slr', StandardScaler()),
            ('fsl', CFS()),
            ('svc', LinearSVC(class_weight='balanced')),
        ],
        'param_grid': [
            {
                'clf__C': CLF_SVC_C,
            },
        ],
    },

}

# analyses
if args.analysis == 1:
    eset_tr_name = 'eset_' + args.dataset_tr
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
        scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
        param_grid=pipelines[args.fs_meth]['param_grid'], error_score=0, return_train_score=False,
        n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X_tr, y_tr)
    joblib.dump(grid, 'data/grid_' + args.dataset_tr + '_' + args.fs_meth.lower() + '.pkl')
    # print selected feature information
    feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
    print(
        'Features: %3s' % feature_idxs.size,
        ' ROC AUC (CV): %.4f' % grid.best_score_,
        ' Params:',  grid.best_params_,
    )
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
    feature_names = feature_names[feature_idxs]
    coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    print('Feature Rankings:')
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
    # plot num top ranked features selected vs roc auc, bcr
    if args.fs_meth in ('Limma-KBest', 'MI-KBest'):
        new_shape = (len(SKB_N_FEATURES), len(CLF_SVC_C))
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__k'])
        )
    elif args.fs_meth in ('Limma-Fpr-SVM-RFE', 'SVM-RFE'):
        new_shape = (len(RFE_N_FEATURES), len(CLF_SVC_C) ** 2)
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__n_features_to_select'])
        )
    elif args.fs_meth == 'SVM-SFM':
        new_shape = (len(SFM_THRESHOLDS), len(SFM_SVC_C) * len(CLF_SVC_C))
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__threshold']).astype(str)
        )
    elif args.fs_meth == 'ExtraTrees-SFM':
        new_shape = (len(SFM_THRESHOLDS), len(CLF_SVC_C))
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__threshold']).astype(str)
        )
    mean_roc_aucs = np.reshape(grid.cv_results_['mean_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    std_roc_aucs = np.reshape(grid.cv_results_['std_test_roc_auc'][xaxis_group_sorted_idxs], new_shape)
    mean_roc_aucs_max_idxs = np.argmax(mean_roc_aucs, axis=1)
    mean_roc_aucs = mean_roc_aucs[np.arange(len(mean_roc_aucs)), mean_roc_aucs_max_idxs]
    std_roc_aucs = std_roc_aucs[np.arange(len(std_roc_aucs)), mean_roc_aucs_max_idxs]
    mean_bcrs = np.reshape(grid.cv_results_['mean_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    std_bcrs = np.reshape(grid.cv_results_['std_test_bcr'][xaxis_group_sorted_idxs], new_shape)
    mean_bcrs_max_idxs = np.argmax(mean_bcrs, axis=1)
    mean_bcrs = mean_bcrs[np.arange(len(mean_bcrs)), mean_bcrs_max_idxs]
    std_bcrs = std_bcrs[np.arange(len(std_bcrs)), mean_bcrs_max_idxs]
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    dataset_name = dataset_name.replace('gse', 'GSE')
    plt.title(
        dataset_name + ' SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected on CV Performance Metrics'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('Cross-validation Score')
    if args.fs_meth in ('Limma-KBest', 'MI-KBest'):
        x_axis = SKB_N_FEATURES
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth in ('Limma-Fpr-SVM-RFE', 'SVM-RFE'):
        x_axis = RFE_N_FEATURES
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth in ('SVM-SFM', 'ExtraTrees-SFM'):
        x_axis = range(len(SFM_THRESHOLDS))
        plt.xticks(x_axis, SFM_THRESHOLDS)
    plt.plot(
        x_axis,
        mean_roc_aucs,
        lw=4, alpha=0.8, label='Mean ROC AUC'
    )
    plt.fill_between(
        x_axis,
        [m - s for m, s in zip(mean_roc_aucs, std_roc_aucs)],
        [m + s for m, s in zip(mean_roc_aucs, std_roc_aucs)],
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.plot(
        x_axis,
        mean_bcrs,
        lw=4, alpha=0.8, label='Mean BCR'
    )
    plt.fill_between(
        x_axis,
        [m - s for m, s in zip(mean_bcrs, std_bcrs)],
        [m + s for m, s in zip(mean_bcrs, std_bcrs)],
        color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
    )
    plt.legend(loc='lower right')
    plt.grid('on')
elif args.analysis == 2:
    eset_tr_name = 'eset_' + args.dataset_tr
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
        scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
        param_grid=pipelines[args.fs_meth]['param_grid'], error_score=0, return_train_score=False,
        n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X_tr, y_tr)
    joblib.dump(grid, 'data/grid_' + args.dataset_tr + '_' + args.fs_meth.lower() + '.pkl')
    # print selected feature information
    feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
    print(
        'Features: %3s' % feature_idxs.size,
        ' ROC AUC (CV): %.4f' % grid.best_score_,
        ' Params:',  grid.best_params_,
    )
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
    feature_names = feature_names[feature_idxs]
    coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    print('Feature Rankings:')
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
    # plot roc curves
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    dataset_name = dataset_name.replace('gse', 'GSE')
    plt.title(
        dataset_name + ' SVM Classifier (' + args.fs_meth + ' ' +
        str(len(feature_idxs)) + ' Features)\nGEO LUAD Test Datasets ROC Curves'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    dataset_te_names = [te_name for _, te_name in dataset_pair_names]
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 500)
    for dataset_te_name in dataset_te_names:
        eset_te_name = 'eset_' + dataset_te_name
        base.load('data/' + eset_te_name + '.Rda')
        eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        X_te = np.array(base.t(biobase.exprs(eset_te)))
        y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
        y_score = grid.decision_function(X_te)
        fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
        roc_auc = roc_auc_score(y_te, y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_aucs.append(roc_auc)
        dataset_te_name = dataset_te_name.replace('gse', 'GSE')
        plt.plot(
            fpr, tpr, lw=4, alpha=0.5,
            label=r'%s ROC (AUC = %0.4f)' % (dataset_te_name, roc_auc),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=4, alpha=0.8,
        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.plot([0,1], [0,1], color='darkred', lw=4, linestyle='--', alpha=0.8, label='Chance')
    plt.legend(loc='lower right')
    plt.grid('off')
elif args.analysis == 3:
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
                Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
                scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
                param_grid=pipelines[args.fs_meth]['param_grid'], error_score=0, return_train_score=False,
                n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if bc_method != 'none':
                joblib.dump(grid, 'data/grid_' + dataset_tr_name + '_' + bc_method + '_' + args.fs_meth.lower() + '.pkl')
            else:
                joblib.dump(grid, 'data/grid_' + dataset_tr_name + '_' + args.fs_meth.lower() + '.pkl')
            # print selected feature information
            feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
            print(
                'Features: %3s' % feature_idxs.size,
                ' ROC AUC (CV): %.4f' % grid.best_score_,
                ' Params:',  grid.best_params_,
            )
            feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
            feature_names = feature_names[feature_idxs]
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            print('Feature Rankings:')
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
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
            y_score = grid.decision_function(X_te)
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc = roc_auc_score(y_te, y_score)
            result = {
                'grid': grid,
                'feature_idxs': feature_idxs,
                'feature_names': feature_names,
                'fprs': fpr,
                'tprs': tpr,
                'thres': thres,
                'coefs': coefs,
                'y_scores': y_score,
                'y_tests': y_te,
                'roc_auc_cv': grid.best_score_,
                'roc_auc_te': roc_auc,
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
    # plot effect bc method vs test dataset roc auc
    plt.figure(3)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on Classifier Performance\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt_fig_x_axis = range(1, len(bc_methods) + 1)
    plt.xticks(plt_fig_x_axis, bc_methods)
    for te_idx, te_bc_results in enumerate(te_results):
        roc_aucs_cv, roc_aucs_te, num_features = [], [], []
        for result in te_bc_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            num_features.append(len(result['feature_idxs']))
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_num_features = np.std(num_features)
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
            #label='%s (Train CV)' % dataset_tr_name,
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
    plt.legend(loc='best')
    plt.grid('on')
    # plot effect test dataset vs bc roc auc
    plt.figure(4)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on Classifier Performance\n' +
        '(' + args.fs_meth + ' Feature Selection Best Scoring Selected Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    for bc_idx, bc_te_results in enumerate(bc_results):
        roc_aucs_cv, roc_aucs_te, num_features = [], [], []
        for result in bc_te_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            num_features.append(len(result['feature_idxs']))
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_num_features = np.std(num_features)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
            #label='%s (Train CV)' % bc_methods[bc_idx],
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                bc_methods[bc_idx],
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
    plt.legend(loc='best')
    plt.grid('on')
elif args.analysis == 4:
    fs_methods = [
        'Limma-KBest',
        #'MI-KBest',
        'Limma-Fpr-SVM-RFE',
        'SVM-RFE',
        #'SVM-SFM',
        #'ExtraTrees-SFM',
        #'Limma-Fpr-CFS',
    ]
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
                Pipeline(pipelines[fs_method]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
                scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
                cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
                param_grid=pipelines[fs_method]['param_grid'], error_score=0, return_train_score=False,
                n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
            )
            grid.fit(X_tr, y_tr)
            if args.bc_meth:
                joblib.dump(grid, 'data/grid_' + dataset_tr_name + '_' + args.bc_meth + '_' + fs_method.lower() + '.pkl')
            else:
                joblib.dump(grid, 'data/grid_' + dataset_tr_name + '_' + fs_method.lower() + '.pkl')
            # print selected feature information
            feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
            print(
                'Features: %3s' % feature_idxs.size,
                ' ROC AUC (CV): %.4f' % grid.best_score_,
                ' Params:',  grid.best_params_,
            )
            feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
            feature_names = feature_names[feature_idxs]
            coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
            print('Feature Rankings:')
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
            base.load('data/' + eset_te_name + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
            X_te = np.array(base.t(biobase.exprs(eset_te)))
            y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
            y_score = grid.decision_function(X_te)
            fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
            roc_auc = roc_auc_score(y_te, y_score)
            result = {
                'grid': grid,
                'feature_idxs': feature_idxs,
                'feature_names': feature_names,
                'fprs': fpr,
                'tprs': tpr,
                'thres': thres,
                'coefs': coefs,
                'y_scores': y_score,
                'y_tests': y_te,
                'roc_auc_cv': grid.best_score_,
                'roc_auc_te': roc_auc,
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
    # plot effect fs method vs test dataset roc auc
    plt.figure(5)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Feature Selection Method on Classifier Performance\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
    )
    plt.xlabel('Feature Selection Method')
    plt.ylabel('ROC AUC')
    plt_fig_x_axis = range(1, len(fs_methods) + 1)
    plt.xticks(plt_fig_x_axis, fs_methods)
    for te_idx, te_fs_results in enumerate(te_results):
        roc_aucs_cv, roc_aucs_te, num_features = [], [], []
        for result in te_fs_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            num_features.append(len(result['feature_idxs']))
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_num_features = np.std(num_features)
        dataset_tr_name, dataset_te_name = dataset_pair_names[te_idx]
        dataset_tr_name = dataset_tr_name.upper()
        dataset_te_name = dataset_te_name.upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
            #label='%s (Train CV)' % dataset_tr_name,
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                dataset_te_name,
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
    plt.legend(loc='best')
    plt.grid('on')
    # plot effect test dataset vs fs roc auc
    plt.figure(6)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on Classifier Performance\n' +
        '(' + args.bc_meth + ' Batch Effect Correction Best Scoring Selected Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    dataset_te_names = [te_name.upper() for _, te_name in dataset_pair_names]
    plt_fig_x_axis = range(1, len(dataset_te_names) + 1)
    plt.xticks(plt_fig_x_axis, dataset_te_names)
    for fs_idx, fs_te_results in enumerate(fs_results):
        roc_aucs_cv, roc_aucs_te, num_features = [], [], []
        for result in fs_te_results:
            roc_aucs_cv.append(result['roc_auc_cv'])
            roc_aucs_te.append(result['roc_auc_te'])
            num_features.append(len(result['feature_idxs']))
        mean_roc_auc_cv = np.mean(roc_aucs_cv)
        mean_roc_auc_te = np.mean(roc_aucs_te)
        mean_num_features = np.mean(num_features)
        std_roc_auc_cv = np.std(roc_aucs_cv)
        std_roc_auc_te = np.std(roc_aucs_te)
        std_num_features = np.std(num_features)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(
            plt_fig_x_axis, roc_aucs_cv,
            lw=4, alpha=0.8, linestyle='--', color=color, markeredgewidth=4, marker='s',
            #label='%s (Train CV)' % fs_methods[fs_idx],
        )
        plt.plot(
            plt_fig_x_axis, roc_aucs_te,
            lw=4, alpha=0.8, color=color, markeredgewidth=4, marker='s',
            label=r'%s (ROC AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' % (
                fs_methods[fs_idx],
                mean_roc_auc_te, std_roc_auc_te,
                mean_num_features, std_num_features,
            )
        )
    plt.legend(loc='best')
    plt.grid('on')

plt.show()
rmtree(cachedir)
