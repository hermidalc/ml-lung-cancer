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

# limma feature scoring function
def limma(X, y):
    f, pv = r_limma(np.transpose(X), y)
    return np.array(f), np.array(pv)

# bcr perf metrics scoring function
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
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help='mean_coefs or mean_roc_aucs')
parser.add_argument('--gscv-splits', type=int, default=30, help='gscv splits')
parser.add_argument('--gscv-size', type=int, default=0.3, help='gscv size')
parser.add_argument('--gscv-jobs', type=int, default=-1, help='gscv parallel jobs')
parser.add_argument('--gscv-verbose', type=int, default=1, help='gscv verbosity')
parser.add_argument('--rfe-step', type=float, default=0.01, help='rfe step')
parser.add_argument('--rfe-verbose', type=int, default=0, help='rfe verbosity')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help='svm algorithm (liblinear or libsvm)')
args = parser.parse_args()

# specify in sort order (needed by code dealing with gridsearch cv_results)
CLF_SVC_C = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
SFM_SVC_C = [ 0.01, 0.1, 1, 10, 100, 1000 ]
SFM_THRESHOLDS = [ 0.01, 0.02, 0.03, 0.04 ]
SKB_N_FEATURES = list(range(1, args.fs_num_max + 1))
RFE_N_FEATURES = list(range(5, args.fs_num_max + 1, 5))

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
    'Limma-Fpr-RFE-SVM': {
        'pipe_steps': [
            ('sfp', SelectFpr(limma, alpha=0.01)),
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
    'SFM-SVM': {
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
    'SFM-ExtraTrees': {
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
}

# analyses
if args.analysis == 1:
    eset_name = 'eset_gse31210'
    base.load('data/' + eset_name + '.Rda')
    eset = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_name])
    X = np.array(base.t(biobase.exprs(eset)))
    y = np.array(r_filter_eset_relapse_labels(eset), dtype=int)
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
        scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
        param_grid=pipelines[args.fs_meth]['param_grid'], error_score=0, return_train_score=False,
        n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X, y)
    joblib.dump(grid, 'data/grid_analysis_' + str(args.analysis) + '.pkl')
    # print selected feature information
    feature_idxs = grid.best_estimator_.named_steps['fsl'].get_support(indices=True)
    print(
        'Features: %3s' % feature_idxs.size,
        ' ROC AUC (CV): %.4f' % grid.best_score_,
        ' Params:',  grid.best_params_,
    )
    feature_names = np.array(biobase.featureNames(eset), dtype=str)
    feature_names = feature_names[feature_idxs]
    coefs = np.square(grid.best_estimator_.named_steps['clf'].coef_[0])
    print('Feature Rankings:')
    for rank, feature, symbol in sorted(
        zip(
            coefs,
            feature_names,
            r_get_gene_symbols(
                eset, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
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
    elif args.fs_meth == 'Limma-Fpr-RFE-SVM':
        new_shape = (len(RFE_N_FEATURES), len(CLF_SVC_C) ** 2)
        xaxis_group_sorted_idxs = np.argsort(
            np.ma.getdata(grid.cv_results_['param_fsl__n_features_to_select'])
        )
    elif args.fs_meth in ('SFM-SVM', 'SFM-ExtraTrees'):
        new_shape = (len(SFM_THRESHOLDS), len(SFM_SVC_C) * len(CLF_SVC_C))
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
    plt.title(
        'GSE31210 SVM Classifier (' + args.fs_meth + ' Feature Selection)\n' +
        'Effect of Number of Top-Ranked Features Selected on CV Performance Metrics'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('Cross-validation Score')
    if args.fs_meth in ('Limma-KBest', 'MI-KBest'):
        x_axis = SKB_N_FEATURES
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth == 'Limma-Fpr-RFE-SVM':
        x_axis = RFE_N_FEATURES
        plt.xlim([ min(x_axis) - 0.5, max(x_axis) + 0.5 ])
        plt.xticks(x_axis)
    elif args.fs_meth in ('SFM-SVM', 'SFM-ExtraTrees'):
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
    eset_tr_name = 'eset_gse31210'
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    eset_te_names = [
        'eset_gse8894',
        'eset_gse30219',
        'eset_gse37745',
    ]
    esets_te = []
    for eset_te_name in eset_te_names:
        base.load('data/' + eset_te_name + '.Rda')
        esets_te.append((
            eset_te_name,
            r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        ))
    grid = GridSearchCV(
        Pipeline(pipelines[args.fs_meth]['pipe_steps'], memory=joblib.Memory(cachedir=cachedir, verbose=0)),
        scoring={ 'roc_auc': 'roc_auc', 'bcr': make_scorer(bcr_score) }, refit='roc_auc',
        cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=args.gscv_size),
        param_grid=pipelines[args.fs_meth]['param_grid'], error_score=0, return_train_score=False,
        n_jobs=args.gscv_jobs, verbose=args.gscv_verbose,
    )
    grid.fit(X, y)
    joblib.dump(grid, 'data/grid_analysis_' + str(args.analysis) + '.pkl')

    # plot roc curves
    plt.figure(5)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 SVM Classifier Vs GEO LUAD Test Datasets ROC Curves\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 500)
    for idx, te_results in enumerate(results):
        te_tprs, te_roc_aucs, te_num_features = [], [], []
        for split in te_results:
            nf_split = sorted(split, key=lambda k: k['roc_auc_te']).pop()
            te_tprs.append(np.interp(mean_fpr, nf_split['fprs'], nf_split['tprs']))
            te_tprs[-1][0] = 0.0
            te_roc_aucs.append(nf_split['roc_auc_te'])
            te_num_features.append(len(nf_split['feature_idxs']))
        te_mean_tpr = np.mean(te_tprs, axis=0)
        te_mean_tpr[-1] = 1.0
        te_mean_roc_auc = np.mean(te_roc_aucs)
        te_std_roc_auc = np.std(te_roc_aucs)
        te_mean_num_features = np.mean(te_num_features)
        te_std_num_features = np.std(te_num_features)
        eset_te_name = eset_te_names[idx].replace('eset_', '').upper()
        plt.plot(
            mean_fpr, te_mean_tpr, lw=4, alpha=0.5,
            label=r'%s Mean ROC (AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' %
            (eset_te_name, te_mean_roc_auc, te_std_roc_auc, te_mean_num_features, te_std_num_features),
        )
        tprs.append(te_mean_tpr)
        tprs[-1][0] = 0.0
        roc_aucs.append(te_mean_roc_auc)
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
    # plot num top ranked features selected vs mean roc auc
    plt.figure(6)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GEO LUAD Test Datasets (' + fs_title + ' FS)\n' +
        'Effect of Number of Top-Ranked Features Selected on ROC AUC'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('ROC AUC')
    max_features = len(results[0][0])
    plt.xlim([0.5, max_features + 0.5])
    plt_fig2_x_axis = range(1, max_features + 1)
    plt.xticks(plt_fig2_x_axis)
    roc_aucs_tr = []
    for idx, te_results in enumerate(results):
        roc_aucs_te = []
        for split in te_results:
            for nf_idx, nf_split in enumerate(split):
                if nf_idx < len(roc_aucs_tr):
                    roc_aucs_tr[nf_idx].append(nf_split['roc_auc_tr'])
                else:
                    roc_aucs_tr.append([nf_split['roc_auc_tr']])
                if nf_idx < len(roc_aucs_te):
                    roc_aucs_te[nf_idx].append(nf_split['roc_auc_te'])
                else:
                    roc_aucs_te.append([nf_split['roc_auc_te']])
        mean_roc_aucs_te, std_roc_aucs_te = [], []
        for nf_idx in range(len(roc_aucs_te)):
            mean_roc_aucs_te.append(np.mean(roc_aucs_te[nf_idx]))
            std_roc_aucs_te.append(np.std(roc_aucs_te[nf_idx]))
        eset_te_name = eset_te_names[idx].replace('eset_', '').upper()
        plt.plot(
            plt_fig2_x_axis, mean_roc_aucs_te,
            lw=4, alpha=0.8, label='%s Mean ROC AUC (Test)' % eset_te_name,
        )
        plt.fill_between(
            plt_fig2_x_axis,
            [m - s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
            [m + s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
            color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
        )
    mean_roc_aucs_tr, std_roc_aucs_tr = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
    plt.plot(
        plt_fig2_x_axis, mean_roc_aucs_tr,
        lw=4, alpha=0.8, label='GSE31210 Mean ROC AUC (Train CV)',
    )
    plt.fill_between(
        plt_fig2_x_axis,
        [m - s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        [m + s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.legend(loc='lower right')
    plt.grid('on')
    # print final selected feature information
    for te_idx, te_results in enumerate(results):
        feature_idxs = []
        for split in te_results:
            nf_split = sorted(split, key=lambda k: k['roc_auc_te']).pop()
            feature_idxs.extend(nf_split['feature_idxs'])
        feature_idxs = sorted(list(set(feature_idxs)))
        feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
        feature_names = feature_names[feature_idxs]
        # print(*natsorted(feature_names), sep='\n')
        feature_mx_idx = {}
        for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
        coef_mx = np.zeros((len(feature_idxs), len(te_results)), dtype=float)
        roc_auc_mx = np.zeros((len(feature_idxs), len(te_results)), dtype=float)
        for split_idx in range(len(te_results)):
            split_data = sorted(te_results[split_idx], key=lambda k: k['roc_auc_te']).pop()
            for idx in range(len(split_data['feature_idxs'])):
                coef_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                    split_data['coefs'][idx]
                roc_auc_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                    split_data['roc_auc_te']
        feature_mean_coefs, feature_mean_roc_aucs = [], []
        for idx in range(len(feature_idxs)):
            feature_mean_coefs.append(np.mean(coef_mx[idx]))
            feature_mean_roc_aucs.append(np.mean(roc_auc_mx[idx]))
            # print(feature_names[idx], '\t', feature_mean_coefs[idx], '\t', coef_mx[idx])
            # print(feature_names[idx], '\t', feature_mean_roc_aucs[idx], '\t', roc_auc_mx[idx])
        if args.fs_rank_meth == 'mean_coefs':
            feature_ranks = feature_mean_coefs
        else:
            feature_ranks = feature_mean_roc_aucs
        eset_te_name = eset_te_names[te_idx].replace('eset_', '').upper()
        print('%s Best Scoring Features:' % eset_te_name)
        for rank, feature, symbol in sorted(
            zip(
                feature_ranks,
                feature_names,
                r_get_gene_symbols(
                    eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                ),
            ),
            reverse=True
        ): print(feature, '\t', symbol, '\t', rank)
elif args.analysis in (7, 8):
    eset_tr_name = 'eset_gse31210'
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    eset_te_names = [
        'eset_gse8894',
        'eset_gse30219',
        'eset_gse37745',
    ]
    esets_te = []
    for eset_te_name in eset_te_names:
        base.load('data/' + eset_te_name + '.Rda')
        esets_te.append((
            eset_te_name,
            r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name])
        ))
    if args.analysis == 7:
        results = pipeline_one_vs_many(eset_tr, esets_te, fs_limma, tr_rfecv_svm)
        fs_title = 'Limma-RFECV'
    elif args.analysis == 8:
        results = pipeline_one_vs_many(eset_tr, esets_te, fs_limma_svm, tr_rfecv_svm)
        fs_title = 'Limma-SVM-RFECV'
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    # plot roc curves
    plt.figure(7)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GEO LUAD Test Datasets ROC Curves\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 500)
    for idx, te_results in enumerate(results):
        te_tprs, te_roc_aucs, te_num_features = [], [], []
        for split in te_results:
            te_tprs.append(np.interp(mean_fpr, split['fprs'], split['tprs']))
            te_tprs[-1][0] = 0.0
            te_roc_aucs.append(split['roc_auc_te'])
            te_num_features.append(len(split['feature_idxs']))
        te_mean_tpr = np.mean(te_tprs, axis=0)
        te_mean_tpr[-1] = 1.0
        te_mean_roc_auc = np.mean(te_roc_aucs)
        te_std_roc_auc = np.std(te_roc_aucs)
        te_mean_num_features = np.mean(te_num_features)
        te_std_num_features = np.std(te_num_features)
        eset_te_name = eset_te_names[idx].replace('eset_', '').upper()
        plt.plot(
            mean_fpr, te_mean_tpr, lw=4, alpha=0.5,
            label=r'%s Mean ROC (AUC = %0.4f $\pm$ %0.2f, Features = %d $\pm$ %d)' %
            (eset_te_name, te_mean_roc_auc, te_std_roc_auc, te_mean_num_features, te_std_num_features),
        )
        tprs.append(te_mean_tpr)
        tprs[-1][0] = 0.0
        roc_aucs.append(te_mean_roc_auc)
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
    # print final selected feature information
    for te_idx, te_results in enumerate(results):
        feature_idxs = []
        for split in te_results: feature_idxs.extend(split['feature_idxs'])
        feature_idxs = sorted(list(set(feature_idxs)))
        feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
        feature_names = feature_names[feature_idxs]
        # print(*natsorted(feature_names), sep='\n')
        feature_mx_idx = {}
        for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
        coef_mx = np.zeros((len(feature_idxs), len(te_results)), dtype=float)
        roc_auc_mx = np.zeros((len(feature_idxs), len(te_results)), dtype=float)
        for split_idx in range(len(te_results)):
            split_data = te_results[split_idx]
            for idx in range(len(split_data['feature_idxs'])):
                coef_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                    split_data['coefs'][idx]
                roc_auc_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                    split_data['roc_auc_te']
        feature_mean_coefs, feature_mean_roc_aucs = [], []
        for idx in range(len(feature_idxs)):
            feature_mean_coefs.append(np.mean(coef_mx[idx]))
            feature_mean_roc_aucs.append(np.mean(roc_auc_mx[idx]))
            # print(feature_names[idx], '\t', feature_mean_coefs[idx], '\t', coef_mx[idx])
            # print(feature_names[idx], '\t', feature_mean_roc_aucs[idx], '\t', roc_auc_mx[idx])
        if args.fs_rank_meth == 'mean_coefs':
            feature_ranks = feature_mean_coefs
        else:
            feature_ranks = feature_mean_roc_aucs
        eset_te_name = eset_te_names[te_idx].replace('eset_', '').upper()
        print('%s Best Scoring Features:' % eset_te_name)
        for rank, feature, symbol in sorted(
            zip(
                feature_ranks,
                feature_names,
                r_get_gene_symbols(
                    eset_tr, robjects.IntVector(np.array(feature_idxs, dtype=int) + 1)
                ),
            ),
            reverse=True
        ): print(feature, '\t', symbol, '\t', rank)
elif args.analysis in (9, 10):
    eset_pair_names = [
        ('eset_gse31210_gse30219_gse37745', 'eset_gse8894'),
        ('eset_gse31210_gse8894_gse37745', 'eset_gse30219'),
        ('eset_gse8894_gse30219_gse37745', 'eset_gse31210'),
        ('eset_gse31210_gse8894_gse30219', 'eset_gse37745'),
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
    te_results, bc_results = [], []
    for te_idx, (eset_tr_name, eset_te_name) in enumerate(eset_pair_names):
        for bc_idx, bc_method in enumerate(bc_methods):
            bc_ext_tr, bc_ext_te = '', ''
            if bc_method != 'none':
                bc_ext_tr, bc_ext_te = '_tr_' + bc_method, '_te_' + bc_method
            print(eset_tr_name + bc_ext_tr, '->', eset_te_name + bc_ext_te)
            base.load('data/' + eset_tr_name + bc_ext_tr + '.Rda')
            eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name + bc_ext_tr])
            base.load('data/' + eset_te_name + bc_ext_te + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name + bc_ext_te])
            if args.analysis == 9:
                results = pipeline_one_vs_one(eset_tr, eset_te, fs_limma, tr_topfwd_svm)
            elif args.analysis == 10:
                results = pipeline_one_vs_one(eset_tr, eset_te, fs_limma_svm, tr_topfwd_svm)
            if te_idx < len(te_results):
                te_results[te_idx].append(results)
            else:
                te_results.append([results])
            if bc_idx < len(bc_results):
                bc_results[bc_idx].append(results)
            else:
                bc_results.append([results])
            base.remove(eset_tr_name + bc_ext_tr)
            base.remove(eset_te_name + bc_ext_te)
    # save results
    all_results = { 'te': te_results, 'bc': bc_results }
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(all_results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    if args.analysis == 9:
        fs_title = 'Limma-TopForward'
    elif args.analysis == 10:
        fs_title = 'Limma-SVM-TopForward'
    # plot effect bc method vs test dataset roc auc
    plt.figure(8)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on Classifier Performance\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt_fig1_x_axis = range(1, len(bc_methods) + 1)
    plt.xticks(plt_fig1_x_axis, bc_methods)
    for te_idx, te_bc_results in enumerate(te_results):
        mean_roc_aucs_tr_bc, range_roc_aucs_tr_bc = [], [[], []]
        mean_roc_aucs_te_bc, range_roc_aucs_te_bc = [], [[], []]
        num_features_te = []
        for results in te_bc_results:
            roc_aucs_tr_bc, roc_aucs_te_bc = [], []
            for split in results:
                nf_split = sorted(split, key=lambda k: k['roc_auc_te']).pop()
                roc_aucs_tr_bc.append(nf_split['roc_auc_tr'])
                roc_aucs_te_bc.append(nf_split['roc_auc_te'])
                num_features_te.append(len(nf_split['feature_idxs']))
            mean_roc_aucs_tr_bc.append(np.mean(roc_aucs_tr_bc))
            range_roc_aucs_tr_bc[0].append(np.mean(roc_aucs_tr_bc) - min(roc_aucs_tr_bc))
            range_roc_aucs_tr_bc[1].append(max(roc_aucs_tr_bc) - np.mean(roc_aucs_tr_bc))
            mean_roc_aucs_te_bc.append(np.mean(roc_aucs_te_bc))
            range_roc_aucs_te_bc[0].append(np.mean(roc_aucs_te_bc) - min(roc_aucs_te_bc))
            range_roc_aucs_te_bc[1].append(max(roc_aucs_te_bc) - np.mean(roc_aucs_te_bc))
        mean_num_features_te = np.mean(num_features_te)
        std_num_features_te = np.std(num_features_te)
        eset_tr_name, eset_te_name = eset_pair_names[te_idx]
        eset_tr_name = eset_tr_name.replace('eset_', '').upper()
        eset_tr_name = eset_tr_name.replace('_', '-')
        eset_te_name = eset_te_name.replace('eset_', '').upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.errorbar(
            plt_fig1_x_axis, mean_roc_aucs_tr_bc, yerr=range_roc_aucs_tr_bc, lw=4, alpha=0.8,
            linestyle='--', capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            #label='%s (Train CV)' % eset_tr_name,
        )
        plt.errorbar(
            plt_fig1_x_axis, mean_roc_aucs_te_bc, yerr=range_roc_aucs_te_bc, lw=4, alpha=0.8,
            capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            label=r'%s (Features = %d $\pm$ %d)' %
            (eset_te_name, mean_num_features_te, std_num_features_te)
        )
    plt.legend(loc='best')
    plt.grid('on')
    # plot effect test dataset vs bc roc auc
    plt.figure(9)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on Classifier Performance\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    eset_te_names = [te_name.replace('eset_', '').upper() for _, te_name in eset_pair_names]
    plt_fig2_x_axis = range(1, len(eset_te_names) + 1)
    plt.xticks(plt_fig2_x_axis, eset_te_names)
    for bc_idx, bc_te_results in enumerate(bc_results):
        mean_roc_aucs_bc_tr, range_roc_aucs_bc_tr = [], [[], []]
        mean_roc_aucs_bc_te, range_roc_aucs_bc_te = [], [[], []]
        num_features_bc = []
        for results in bc_te_results:
            roc_aucs_bc_tr, roc_aucs_bc_te = [], []
            for split in results:
                nf_split = sorted(split, key=lambda k: k['roc_auc_te']).pop()
                roc_aucs_bc_tr.append(nf_split['roc_auc_tr'])
                roc_aucs_bc_te.append(nf_split['roc_auc_te'])
                num_features_bc.append(len(nf_split['feature_idxs']))
            mean_roc_aucs_bc_tr.append(np.mean(roc_aucs_bc_tr))
            range_roc_aucs_bc_tr[0].append(np.mean(roc_aucs_bc_tr) - min(roc_aucs_bc_tr))
            range_roc_aucs_bc_tr[1].append(max(roc_aucs_bc_tr) - np.mean(roc_aucs_bc_tr))
            mean_roc_aucs_bc_te.append(np.mean(roc_aucs_bc_te))
            range_roc_aucs_bc_te[0].append(np.mean(roc_aucs_bc_te) - min(roc_aucs_bc_te))
            range_roc_aucs_bc_te[1].append(max(roc_aucs_bc_te) - np.mean(roc_aucs_bc_te))
        mean_num_features_bc = np.mean(num_features_bc)
        std_num_features_bc = np.std(num_features_bc)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.errorbar(
            plt_fig2_x_axis, mean_roc_aucs_bc_tr, yerr=range_roc_aucs_bc_tr, lw=4, alpha=0.8,
            linestyle='--', capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            #label='%s (Train CV)' % bc_methods[bc_idx],
        )
        plt.errorbar(
            plt_fig2_x_axis, mean_roc_aucs_bc_te, yerr=range_roc_aucs_bc_te, lw=4, alpha=0.8,
            capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            label=r'%s (Features = %d $\pm$ %d)' %
            (bc_methods[bc_idx], mean_num_features_bc, std_num_features_bc)
        )
    plt.legend(loc='best')
    plt.grid('on')
elif args.analysis in (11, 12):
    eset_pair_names = [
        ('eset_gse31210_gse30219_gse37745', 'eset_gse8894'),
        ('eset_gse31210_gse8894_gse37745', 'eset_gse30219'),
        ('eset_gse8894_gse30219_gse37745', 'eset_gse31210'),
        ('eset_gse31210_gse8894_gse30219', 'eset_gse37745'),
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
    te_results, bc_results = [], []
    for te_idx, (eset_tr_name, eset_te_name) in enumerate(eset_pair_names):
        for bc_idx, bc_method in enumerate(bc_methods):
            bc_ext_tr, bc_ext_te = '', ''
            if bc_method != 'none':
                bc_ext_tr, bc_ext_te = '_tr_' + bc_method, '_te_' + bc_method
            print(eset_tr_name + bc_ext_tr, '->', eset_te_name + bc_ext_te)
            base.load('data/' + eset_tr_name + bc_ext_tr + '.Rda')
            eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name + bc_ext_tr])
            base.load('data/' + eset_te_name + bc_ext_te + '.Rda')
            eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name + bc_ext_te])
            if args.analysis == 11:
                results = pipeline_one_vs_one(eset_tr, eset_te, fs_limma, tr_rfecv_svm)
            elif args.analysis == 12:
                results = pipeline_one_vs_one(eset_tr, eset_te, fs_limma_svm, tr_rfecv_svm)
            if te_idx < len(te_results):
                te_results[te_idx].append(results)
            else:
                te_results.append([results])
            if bc_idx < len(bc_results):
                bc_results[bc_idx].append(results)
            else:
                bc_results.append([results])
            base.remove(eset_tr_name + bc_ext_tr)
            base.remove(eset_te_name + bc_ext_te)
    # save results
    all_results = { 'te': te_results, 'bc': bc_results }
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(all_results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    if args.analysis == 11:
        fs_title = 'Limma-TopForward'
    elif args.analysis == 12:
        fs_title = 'Limma-SVM-TopForward'
    # plot effect bc method vs test dataset roc auc
    plt.figure(10)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Batch Effect Correction Method on Classifier Performance\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('Batch Effect Correction Method')
    plt.ylabel('ROC AUC')
    plt_fig1_x_axis = range(1, len(bc_methods) + 1)
    plt.xticks(plt_fig1_x_axis, bc_methods)
    for te_idx, te_bc_results in enumerate(te_results):
        mean_roc_aucs_tr_bc, range_roc_aucs_tr_bc = [], [[], []]
        mean_roc_aucs_te_bc, range_roc_aucs_te_bc = [], [[], []]
        num_features_te = []
        for results in te_bc_results:
            roc_aucs_tr_bc, roc_aucs_te_bc = [], []
            for split in results:
                roc_aucs_tr_bc.append(split['roc_auc_tr'])
                roc_aucs_te_bc.append(split['roc_auc_te'])
                num_features_te.append(len(split['feature_idxs']))
            mean_roc_aucs_tr_bc.append(np.mean(roc_aucs_tr_bc))
            range_roc_aucs_tr_bc[0].append(np.mean(roc_aucs_tr_bc) - min(roc_aucs_tr_bc))
            range_roc_aucs_tr_bc[1].append(max(roc_aucs_tr_bc) - np.mean(roc_aucs_tr_bc))
            mean_roc_aucs_te_bc.append(np.mean(roc_aucs_te_bc))
            range_roc_aucs_te_bc[0].append(np.mean(roc_aucs_te_bc) - min(roc_aucs_te_bc))
            range_roc_aucs_te_bc[1].append(max(roc_aucs_te_bc) - np.mean(roc_aucs_te_bc))
        mean_num_features_te = np.mean(num_features_te)
        std_num_features_te = np.std(num_features_te)
        eset_tr_name, eset_te_name = eset_pair_names[te_idx]
        eset_tr_name = eset_tr_name.replace('eset_', '').upper()
        eset_tr_name = eset_tr_name.replace('_', '-')
        eset_te_name = eset_te_name.replace('eset_', '').upper()
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.errorbar(
            plt_fig1_x_axis, mean_roc_aucs_tr_bc, yerr=range_roc_aucs_tr_bc, lw=4, alpha=0.8,
            linestyle='--', capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            #label='%s (Train CV)' % eset_tr_name,
        )
        plt.errorbar(
            plt_fig1_x_axis, mean_roc_aucs_te_bc, yerr=range_roc_aucs_te_bc, lw=4, alpha=0.8,
            capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            label=r'%s (Features = %d $\pm$ %d)' %
            (eset_te_name, mean_num_features_te, std_num_features_te)
        )
    plt.legend(loc='best')
    plt.grid('on')
    # plot effect test dataset vs bc roc auc
    plt.figure(11)
    plt.rcParams['font.size'] = 20
    plt.title(
        'Effect of Training/Held-Out Test Dataset on Classifier Performance\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('Test Dataset')
    plt.ylabel('ROC AUC')
    eset_te_names = [te_name.replace('eset_', '').upper() for _, te_name in eset_pair_names]
    plt_fig2_x_axis = range(1, len(eset_te_names) + 1)
    plt.xticks(plt_fig2_x_axis, eset_te_names)
    for bc_idx, bc_te_results in enumerate(bc_results):
        mean_roc_aucs_bc_tr, range_roc_aucs_bc_tr = [], [[], []]
        mean_roc_aucs_bc_te, range_roc_aucs_bc_te = [], [[], []]
        num_features_bc = []
        for results in bc_te_results:
            roc_aucs_bc_tr, roc_aucs_bc_te = [], []
            for split in results:
                roc_aucs_bc_tr.append(split['roc_auc_tr'])
                roc_aucs_bc_te.append(split['roc_auc_te'])
                num_features_bc.append(len(split['feature_idxs']))
            mean_roc_aucs_bc_tr.append(np.mean(roc_aucs_bc_tr))
            range_roc_aucs_bc_tr[0].append(np.mean(roc_aucs_bc_tr) - min(roc_aucs_bc_tr))
            range_roc_aucs_bc_tr[1].append(max(roc_aucs_bc_tr) - np.mean(roc_aucs_bc_tr))
            mean_roc_aucs_bc_te.append(np.mean(roc_aucs_bc_te))
            range_roc_aucs_bc_te[0].append(np.mean(roc_aucs_bc_te) - min(roc_aucs_bc_te))
            range_roc_aucs_bc_te[1].append(max(roc_aucs_bc_te) - np.mean(roc_aucs_bc_te))
        mean_num_features_bc = np.mean(num_features_bc)
        std_num_features_bc = np.std(num_features_bc)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.errorbar(
            plt_fig2_x_axis, mean_roc_aucs_bc_tr, yerr=range_roc_aucs_bc_tr, lw=4, alpha=0.8,
            linestyle='--', capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            #label='%s (Train CV)' % bc_methods[bc_idx],
        )
        plt.errorbar(
            plt_fig2_x_axis, mean_roc_aucs_bc_te, yerr=range_roc_aucs_bc_te, lw=4, alpha=0.8,
            capsize=25, elinewidth=4, markeredgewidth=4, marker='s', color=color,
            label=r'%s (Features = %d $\pm$ %d)' %
            (bc_methods[bc_idx], mean_num_features_bc, std_num_features_bc)
        )
    plt.legend(loc='best')
    plt.grid('on')

plt.show()
rmtree(cachedir)
