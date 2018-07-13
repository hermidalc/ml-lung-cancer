#!/usr/bin/env python

import argparse, math, pickle, pprint
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import numpy2ri
# import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from matplotlib import style

base = importr('base')
biobase = importr('Biobase')
base.source('lib/R/functions.R')
r_rand_perm_sample_nums = robjects.globalenv['randPermSampleNums']
r_filter_eset = robjects.globalenv['filterEset']
r_filter_eset_ctrl_probesets = robjects.globalenv['filterEsetControlProbesets']
r_get_eset_class_labels = robjects.globalenv['getEsetClassLabels']
r_get_gene_symbols = robjects.globalenv['getEsetGeneSymbols']
r_get_limma_features = robjects.globalenv['getLimmaFeatures']
# config
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=int, help='analysis run number')
parser.add_argument('--splits', type=int, default=100, help='num splits')
parser.add_argument('--fs-size', type=float, default=0.5, help='fs size')
parser.add_argument('--fs-dfx-max', type=int, default=np.inf, help='fs max num dfx features')
parser.add_argument('--fs-dfx-pval', type=float, default=0.05, help='fs min dfx adj p-value')
parser.add_argument('--fs-dfx-lfc', type=float, default=0, help='fs min dfx logfc')
parser.add_argument('--fs-dfx-select', type=int, default=30, help='fs dfx top select')
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help='mean_coefs or mean_roc_aucs')
parser.add_argument('--fs-final-select', type=int, default=20, help='fs final top select')
parser.add_argument('--fs-gscv-splits', type=int, default=50, help='num fs gscv splits')
parser.add_argument('--fs-gscv-size', type=int, default=0.3, help='fs gscv cv size')
parser.add_argument('--fs-gscv-jobs', type=int, default=-1, help='fs gscv parallel jobs')
parser.add_argument('--fs-gscv-verbose', type=int, default=0, help='gscv verbosity')
parser.add_argument('--tr-gscv-splits', type=int, default=50, help='num tr gscv splits')
parser.add_argument('--tr-gscv-size', type=int, default=0.3, help='tr gscv size')
parser.add_argument('--tr-gscv-jobs', type=int, default=-1, help='tr gscv parallel jobs')
parser.add_argument('--tr-gscv-verbose', type=int, default=0, help='tr gscv verbosity')
parser.add_argument('--tr-rfecv-splits', type=int, default=32, help='num tr rfecv splits')
parser.add_argument('--tr-rfecv-size', type=int, default=0.3, help='rfecv cv size')
parser.add_argument('--tr-rfecv-jobs', type=int, default=-1, help='num tr rfecv parallel jobs')
parser.add_argument('--tr-rfecv-step', type=float, default=1, help='tr rfecv step')
parser.add_argument('--tr-rfecv-verbose', type=int, default=0, help='tr rfecv verbosity')
parser.add_argument('--te-size', type=float, default=0.3, help='te size')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='libsvm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help='svm algorithm (liblinear or libsvm)')
parser.add_argument('--eset-tr', type=str, help='R eset for fs/tr')
parser.add_argument('--eset-te', type=str, help='R eset for te')
args = parser.parse_args()

def pipeline_one(eset, fs_meth, tr_meth):
    X = np.array(base.t(biobase.exprs(eset)))
    y = np.array(r_get_eset_class_labels(eset), dtype=int)
    results = []
    split_count = 0
    fs_fail_count = 0
    print_header = True
    while split_count < args.splits:
        tr_fs_idxs, te_idxs = train_test_split(np.arange(y.size), test_size=args.te_size, stratify=y)
        tr_idxs, fs_idxs = train_test_split(tr_fs_idxs, test_size=args.fs_size, stratify=y[tr_fs_idxs])
        if print_header:
            print('FS: %3s' % fs_idxs.size, ' TR: %3s' % tr_idxs.size, ' TE: %3s' % te_idxs.size)
            print_header = False
        fs_data = fs_meth(
            X[fs_idxs], y[fs_idxs], r_filter_eset(eset, robjects.NULL, robjects.IntVector(fs_idxs + 1))
        )
        if fs_data:
            split_results = tr_meth(X[tr_idxs], y[tr_idxs], X[te_idxs], y[te_idxs], eset, fs_data)
            results.append(split_results)
            split_count += 1
            print('Split: %3s' % split_count, ' Fails: %3s' % fs_fail_count)
        else:
            fs_fail_count += 1
    return(results)
# end pipeline_one

def pipeline_one_vs_many(eset_tr, esets_te, fs_meth, tr_meth):
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_get_eset_class_labels(eset_tr), dtype=int)
    Xys_te = []
    for (eset_te_name, eset_te) in esets_te:
        Xys_te.append((
            eset_te_name,
            np.array(base.t(biobase.exprs(eset_te))),
            np.array(r_get_eset_class_labels(eset_te), dtype=int)
        ))
    results = []
    split_count = 0
    fs_fail_count = 0
    print_header = True
    while split_count < args.splits:
        tr_idxs, fs_idxs = train_test_split(np.arange(y_tr.size), test_size=args.fs_size, stratify=y_tr)
        if print_header:
            print('FS: %3s' % fs_idxs.size, ' TR: %3s' % tr_idxs.size)
            print_header = False
        fs_data = fs_meth(
            X_tr[fs_idxs], y_tr[fs_idxs], r_filter_eset(eset_tr, robjects.NULL, robjects.IntVector(fs_idxs + 1))
        )
        if fs_data:
            for idx, (eset_te_name, X_te, y_te) in enumerate(Xys_te):
                print('TE: %3s [%s]' % (y_te.size, eset_te_name))
                split_results = tr_meth(X_tr[tr_idxs], y_tr[tr_idxs], X_te, y_te, eset_tr, fs_data)
                if idx < len(results):
                    results[idx].append(split_results)
                else:
                    results.append([split_results])
            split_count += 1
            print('Split: %3s' % split_count, ' Fails: %3s' % fs_fail_count)
        else:
            fs_fail_count += 1
    return(results)
# end pipeline_one_vs_many

def pipeline_one_vs_one(eset_tr, eset_te, fs_meth, tr_meth):
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_get_eset_class_labels(eset_tr), dtype=int)
    X_te = np.array(base.t(biobase.exprs(eset_te)))
    y_te = np.array(r_get_eset_class_labels(eset_te), dtype=int)
    results = []
    split_count = 0
    fs_fail_count = 0
    print_header = True
    while split_count < args.splits:
        tr_idxs, fs_idxs = train_test_split(np.arange(y_tr.size), test_size=args.fs_size, stratify=y_tr)
        if print_header:
            print('FS: %3s' % fs_idxs.size, ' TR: %3s' % tr_idxs.size, ' TE: %3s' % y_te.size)
            print_header = False
        fs_data = fs_meth(
            X_tr[fs_idxs], y_tr[fs_idxs], r_filter_eset(eset_tr, robjects.NULL, robjects.IntVector(fs_idxs + 1))
        )
        if fs_data:
            split_results = tr_meth(X_tr[tr_idxs], y_tr[tr_idxs], X_te, y_te, eset_tr, fs_data)
            results.append(split_results)
            split_count += 1
            print('Split: %3s' % split_count, ' Fails: %3s' % fs_fail_count)
        else:
            fs_fail_count += 1
    return(results)
# end pipeline_one_vs_one

def tr_topfwd_svm(X_tr, y_tr, X_te, y_te, eset_tr, fs_data):
    tr_gscv_clf = GridSearchCV(
        Pipeline([
            ('slr', StandardScaler()),
            ('svc', LinearSVC(class_weight='balanced')),
        ]),
        param_grid=[
            { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        ],
        cv=StratifiedShuffleSplit(n_splits=args.tr_gscv_splits, test_size=args.tr_gscv_size),
        scoring='roc_auc', return_train_score=False, n_jobs=args.tr_gscv_jobs,
        verbose=args.tr_gscv_verbose
    )
    results = {
        'fs_data': fs_data,
        'nf_split_data': [],
    }
    feature_idxs = fs_data['feature_idxs']
    feature_names = fs_data['feature_names']
    nf_split_count = 0
    for num_features in range(1, feature_idxs.size + 1):
        top_feature_idxs = feature_idxs[:num_features]
        top_feature_names = feature_names[:num_features]
        y_score = tr_gscv_clf.fit(X_tr[:,top_feature_idxs], y_tr).decision_function(X_te[:,top_feature_idxs])
        fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
        roc_auc = roc_auc_score(y_te, y_score)
        coefs = np.square(tr_gscv_clf.best_estimator_.named_steps['svc'].coef_[0])
        results['nf_split_data'].append({
            'gscv_clf': tr_gscv_clf,
            'feature_idxs': top_feature_idxs,
            'feature_names': top_feature_names,
            'fprs': fpr,
            'tprs': tpr,
            'thres': thres,
            'coefs': coefs,
            'y_scores': y_score,
            'y_tests': y_te,
            'roc_auc_tr': tr_gscv_clf.best_score_,
            'roc_auc_te': roc_auc,
        })
        nf_split_count += 1
        print(
            'Features: %3s' % nf_split_count,
            ' ROC AUC (Train): %.4f' % tr_gscv_clf.best_score_,
            ' ROC AUC (Test): %.4f' % roc_auc,
        )
    # end for
    # best_result = sorted(results, key=lambda k: k['roc_auc_te']).pop()
    # print('Features:', best_result['feature_idxs'].size)
    # for rank, feature, symbol in sorted(
    #     zip(
    #         best_result['coefs'],
    #         best_result['feature_names'],
    #         r_get_gene_symbols(eset_tr, robjects.IntVector(best_result['feature_idxs'] + 1)),
    #     ),
    #     reverse=True
    # ): print(feature, '\t', symbol, '\t', rank)
    return(results)
# end tr_topfwd_svm

def tr_rfecv_svm(X_tr, y_tr, X_te, y_te, eset_tr, fs_data):
    tr_gscv_clf = GridSearchCV(
        Pipeline([
            ('slr', StandardScaler()),
            ('rfe', RFECV(
                LinearSVC(class_weight='balanced'), step=args.tr_rfecv_step,
                cv=StratifiedShuffleSplit(n_splits=args.tr_rfecv_splits, test_size=args.tr_rfecv_size),
                scoring='roc_auc', n_jobs=args.tr_rfecv_jobs, verbose=args.tr_rfecv_verbose
            )),
            ('svc', LinearSVC(class_weight='balanced')),
        ]),
        param_grid=[
            { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        ],
        cv=StratifiedShuffleSplit(n_splits=args.tr_gscv_splits, test_size=args.tr_gscv_size),
        scoring='roc_auc', return_train_score=False, verbose=args.tr_gscv_verbose
    )
    feature_idxs = fs_data['feature_idxs']
    feature_names = fs_data['feature_names']
    y_score = tr_gscv_clf.fit(X_tr[:,feature_idxs], y_tr).decision_function(X_te[:,feature_idxs])
    fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
    roc_auc = roc_auc_score(y_te, y_score)
    rfe_feature_idxs = feature_idxs[tr_gscv_clf.best_estimator_.named_steps['rfe'].get_support(indices=True)]
    rfe_feature_names = feature_names[tr_gscv_clf.best_estimator_.named_steps['rfe'].get_support(indices=True)]
    coefs = np.square(tr_gscv_clf.best_estimator_.named_steps['svc'].coef_[0])
    results = {
        'fs_data': fs_data,
        'gscv_clf': tr_gscv_clf,
        'feature_idxs': rfe_feature_idxs,
        'feature_names': rfe_feature_names,
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': coefs,
        'y_scores': y_score,
        'y_tests': y_te,
        'roc_auc_tr': tr_gscv_clf.best_score_,
        'roc_auc_te': roc_auc,
    }
    print('Num Features:', tr_gscv_clf.best_estimator_.named_steps['rfe'].n_features_)
    for rank, feature, symbol in sorted(
        zip(
            coefs,
            rfe_feature_names,
            r_get_gene_symbols(eset_tr, robjects.IntVector(rfe_feature_idxs + 1)),
        ),
        reverse=True
    ): print(feature, '\t', symbol, '\t', rank)
    print(
        'ROC AUC (Train): %.4f' % tr_gscv_clf.best_score_,
        ' ROC AUC (Test): %.4f' % roc_auc,
    )
    return(results)
# end tr_rfecv_svm

def fs_limma(X_fs, y_fs, eset_fs):
    feature_idxs = np.array(
        r_get_limma_features(
            eset_fs,
            True,
            args.fs_dfx_pval,
            args.fs_dfx_lfc,
            args.fs_dfx_max,
        )
    ) - 1
    if feature_idxs.size < args.fs_dfx_select: return()
    feature_names = np.array(biobase.featureNames(eset_fs), dtype=str)
    fs_num_features = min(args.fs_dfx_select, len(feature_idxs))
    fs_data = {
        'feature_idxs': feature_idxs[:fs_num_features],
        'feature_names': feature_names[feature_idxs[:fs_num_features]],
    }
    print('Features: %3s / %3s' % (fs_num_features, len(feature_idxs)))
    return(fs_data)
# end fs limma

def fs_limma_svm(X_fs, y_fs, eset_fs):
    feature_idxs = np.array(
        r_get_limma_features(
            eset_fs,
            True,
            args.fs_dfx_pval,
            args.fs_dfx_lfc,
            args.fs_dfx_max,
        )
    ) - 1
    if feature_idxs.size < args.fs_dfx_select: return()
    fs_gscv_clf = GridSearchCV(
        Pipeline([
            ('slr', StandardScaler()),
            ('svc', LinearSVC(class_weight='balanced')),
        ]),
        param_grid=[
            { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        ],
        cv=StratifiedShuffleSplit(n_splits=args.fs_gscv_splits, test_size=args.fs_gscv_size),
        scoring='roc_auc', return_train_score=False, n_jobs=args.fs_gscv_jobs,
        verbose=args.fs_gscv_verbose
    )
    fs_gscv_clf.fit(X_fs[:,feature_idxs], y_fs)
    coefs = np.square(fs_gscv_clf.best_estimator_.named_steps['svc'].coef_[0])
    feature_names = np.array(biobase.featureNames(eset_fs), dtype=str)
    feature_rank_data = sorted(zip(coefs, feature_idxs, feature_names[feature_idxs]), reverse=True)
    fs_num_features = min(args.fs_dfx_select, len(feature_idxs))
    fs_data = {
        'coefs': np.array([x for x, _, _ in feature_rank_data[:fs_num_features]]),
        'feature_idxs': np.array([x for _, x, _ in feature_rank_data[:fs_num_features]], dtype=int),
        'feature_names': np.array([x for _, _, x in feature_rank_data[:fs_num_features]], dtype=str),
        'feature_rank_data': feature_rank_data,
    }
    print(
        'Features: %3s / %3s' % (fs_num_features, len(feature_idxs)),
        ' ROC AUC: %.4f' % fs_gscv_clf.best_score_
    )
    return(fs_data)
# end fs limma svm

# analyses
if args.analysis in (1, 2):
    eset_tr_name = 'eset_gse31210'
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    if args.analysis == 1:
        results = pipeline_one(eset_tr, fs_limma, tr_topfwd_svm)
        fs_title = 'Limma-TopForward'
    elif args.analysis == 2:
        results = pipeline_one(eset_tr, fs_limma_svm, tr_topfwd_svm)
        fs_title = 'Limma-SVM-TopForward'
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    # plot roc curves
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GSE31210 Test ROC Curves\n' + fs_title +
        ' Feature Selection (Top ' + str(args.fs_final_select) + ' Ranked Features)'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 500)
    for idx, split in enumerate(results):
        nf_split = split['nf_split_data'][args.fs_final_select - 1]
        tprs.append(np.interp(mean_fpr, nf_split['fprs'], nf_split['tprs']))
        tprs[-1][0] = 0.0
        roc_aucs.append(nf_split['roc_auc_te'])
        plt.plot(
            nf_split['fprs'], nf_split['tprs'], lw=2, alpha=0.3,
            # label='ROC split %d (AUC = %0.4f)' % (idx + 1, nf_split['roc_auc_te']),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=4, alpha=0.8,
        label=r'Test Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
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
    roc_aucs_tr, roc_aucs_te = [], []
    for split in results:
        for nf_idx, nf_split in enumerate(split['nf_split_data']):
            if nf_idx < len(roc_aucs_tr):
                roc_aucs_tr[nf_idx].append(nf_split['roc_auc_tr'])
                roc_aucs_te[nf_idx].append(nf_split['roc_auc_te'])
            else:
                roc_aucs_tr.append([nf_split['roc_auc_tr']])
                roc_aucs_te.append([nf_split['roc_auc_te']])
    mean_roc_aucs_tr, std_roc_aucs_tr = [], []
    mean_roc_aucs_te, std_roc_aucs_te = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        mean_roc_aucs_te.append(np.mean(roc_aucs_te[nf_idx]))
        std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
        std_roc_aucs_te.append(np.std(roc_aucs_te[nf_idx]))
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GSE31210 Test (' + fs_title + ' FS)\n' +
        'Effect of Number of Top-Ranked Features Selected on ROC AUC'
    )
    plt.xlabel('Number of top-ranked features selected')
    plt.ylabel('ROC AUC')
    plt.xlim([0.5, len(mean_roc_aucs_tr) + 0.5])
    plt_fig2_x_axis = range(1, len(mean_roc_aucs_tr) + 1)
    plt.xticks(plt_fig2_x_axis)
    plt.plot(
        plt_fig2_x_axis, mean_roc_aucs_tr,
        lw=4, alpha=0.8, label='Mean ROC AUC (Train CV)',
        # label=r'Train Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (roc_auc_tr_mean, roc_auc_tr_std),
    )
    plt.fill_between(
        plt_fig2_x_axis,
        [m - s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        [m + s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.plot(
        plt_fig2_x_axis, mean_roc_aucs_te,
        lw=4, alpha=0.8, label='Mean ROC AUC (Test)',
        # label=r'Test Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (roc_auc_te_mean, roc_auc_te_std),
    )
    plt.fill_between(
        plt_fig2_x_axis,
        [m - s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
        [m + s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
        color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
    )
    plt.legend(loc='lower right')
    plt.grid('on')
    # print final selected feature information
    feature_idxs = []
    for split in results:
        feature_idxs.extend(split['nf_split_data'][args.fs_final_select - 1]['feature_idxs'])
    feature_idxs = sorted(list(set(feature_idxs)))
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
    feature_names = feature_names[feature_idxs]
    # print(*natsorted(feature_names), sep='\n')
    feature_mx_idx = {}
    for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
    coef_mx = np.zeros((len(feature_idxs), len(results)), dtype=float)
    roc_auc_mx = np.zeros((len(feature_idxs), len(results)), dtype=float)
    for split_idx in range(len(results)):
        split_data = results[split_idx]['nf_split_data'][args.fs_final_select - 1]
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
    print('Top Classifier Features:')
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
elif args.analysis in (3, 4):
    eset_tr_name = 'eset_gse31210'
    base.load('data/' + eset_tr_name + '.Rda')
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    if args.analysis == 3:
        results = pipeline_one(eset_tr, fs_limma, tr_rfecv_svm)
        fs_title = 'Limma-RFECV'
    elif args.analysis == 4:
        results = pipeline_one(eset_tr, fs_limma_svm, tr_rfecv_svm)
        fs_title = 'Limma-SVM-RFECV'
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    # plot roc curves
    plt.figure(3)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GSE31210 Test ROC Curves\n' +
        fs_title + ' Feature Selection (Best Scoring Number of Features)'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs, num_features = [], [], []
    mean_fpr = np.linspace(0, 1, 500)
    for idx, split in enumerate(results):
        tprs.append(np.interp(mean_fpr, split['fprs'], split['tprs']))
        tprs[-1][0] = 0.0
        roc_aucs.append(split['roc_auc_te'])
        num_features.append(len(split['feature_idxs']))
        plt.plot(
            split['fprs'], split['tprs'], lw=2, alpha=0.3,
            # label='ROC split %d (AUC = %0.4f)' % (idx + 1, split['roc_auc_te']),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    mean_num_features = np.mean(num_features)
    std_num_features = np.std(num_features)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=4, alpha=0.8,
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
    plt.plot([0,1], [0,1], color='darkred', lw=4, linestyle='--', alpha=0.8, label='Chance')
    plt.legend(loc='lower right')
    plt.grid('off')
    # plot num features selected vs train roc auc
    roc_aucs_tr = []
    for split in results:
        for nf_idx, roc_auc_tr in enumerate(
            split['gscv_clf'].best_estimator_.named_steps['rfe'].grid_scores_
        ):
            if nf_idx < len(roc_aucs_tr):
                roc_aucs_tr[nf_idx].append(roc_auc_tr)
            else:
                roc_aucs_tr.append([roc_auc_tr])
    mean_roc_aucs_tr, std_roc_aucs_tr = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
    plt.figure(4)
    plt.rcParams['font.size'] = 20
    plt.title(
        'GSE31210 Train SVM Classifier Vs GSE31210 Test (' + fs_title + ' FS)\n' +
        'Effect of Number of RFECV Features Selected on ROC AUC'
    )
    plt.xlabel('Number of features selected')
    plt.ylabel('ROC AUC')
    plt.ylim(0.45, 0.9)
    plt_fig2_x_axis = range(1, len(roc_aucs_tr) + 1)
    plt.xlim([0.5, len(roc_aucs_tr) + 0.5])
    plt.xticks(plt_fig2_x_axis)
    plt.plot(
        plt_fig2_x_axis, mean_roc_aucs_tr,
        lw=4, alpha=0.8, label='Mean ROC AUC (Train CV)',
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
    feature_idxs = []
    for split in results: feature_idxs.extend(split['feature_idxs'])
    feature_idxs = sorted(list(set(feature_idxs)))
    feature_names = np.array(biobase.featureNames(eset_tr), dtype=str)
    feature_names = feature_names[feature_idxs]
    # print(*natsorted(feature_names), sep='\n')
    feature_mx_idx = {}
    for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
    coef_mx = np.zeros((len(feature_idxs), len(results)), dtype=float)
    roc_auc_mx = np.zeros((len(feature_idxs), len(results)), dtype=float)
    for split_idx in range(len(results)):
        split_data = results[split_idx]
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
    print('Top Classifier Features:')
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
elif args.analysis in (5, 6):
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
    if args.analysis == 5:
        results = pipeline_one_vs_many(eset_tr, esets_te, fs_limma, tr_topfwd_svm)
        fs_title = 'Limma-TopForward'
    elif args.analysis == 6:
        results = pipeline_one_vs_many(eset_tr, esets_te, fs_limma_svm, tr_topfwd_svm)
        fs_title = 'Limma-SVM-TopForward'
    results_fh = open('data/results_analysis_' + str(args.analysis) + '.pkl', 'wb')
    pickle.dump(results, results_fh, pickle.HIGHEST_PROTOCOL)
    results_fh.close()
    # plot roc curves
    plt.figure(5)
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
            nf_split = sorted(split['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
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
    max_features = len(results[0][0]['nf_split_data'])
    plt.xlim([0.5, max_features + 0.5])
    plt_fig2_x_axis = range(1, max_features + 1)
    plt.xticks(plt_fig2_x_axis)
    roc_aucs_tr = []
    for idx, te_results in enumerate(results):
        roc_aucs_te = []
        for split in te_results:
            for nf_idx, nf_split in enumerate(split['nf_split_data']):
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
            nf_split = sorted(split['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
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
            split_data = sorted(te_results[split_idx]['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
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
                nf_split = sorted(split['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
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
                nf_split = sorted(split['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
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
