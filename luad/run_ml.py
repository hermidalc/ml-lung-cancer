#!/usr/bin/env python

import argparse, math, pickle, pprint
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import numpy2ri
# import pandas as pd
import numpy as np
from natsort import natsorted
from scipy import interp
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.externals import joblib
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
base = importr("base")
biobase = importr("Biobase")
base.source("functions.R")
r_rand_perm_sample_nums = robjects.globalenv["randPermSampleNums"]
r_filter_eset = robjects.globalenv["filterEset"]
r_filter_eset_ctrl_probesets = robjects.globalenv["filterEsetControlProbesets"]
r_filter_eset_relapse_labels = robjects.globalenv["filterEsetRelapseLabels"]
r_get_gene_symbols = robjects.globalenv["getGeneSymbols"]
r_get_dfx_features = robjects.globalenv["getDfxFeatures"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, help="run number")
parser.add_argument('--fs-splits', type=int, default=100, help='num fs splits')
parser.add_argument('--fs-cv-size', type=float, default=0.3, help="fs cv size")
parser.add_argument('--fs-dfx-min', type=int, default=10, help='fs min num dfx features')
parser.add_argument('--fs-dfx-max', type=int, default=50, help='fs max num dfx features')
parser.add_argument('--fs-dfx-pval', type=float, default=0.01, help="min dfx adj p value")
parser.add_argument('--fs-dfx-lfc', type=float, default=0, help="min dfx logfc")
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help="mean_coefs or mean_roc_aucs")
parser.add_argument('--fs-top-cutoff', type=int, default=30, help='fs top ranked features cutoff')
parser.add_argument('--fs-gscv-splits', type=int, default=50, help='num fs gscv splits')
parser.add_argument('--fs-gscv-size', type=int, default=0.3, help='fs gscv cv size')
parser.add_argument('--fs-gscv-jobs', type=int, default=-1, help="fs gscv parallel jobs")
parser.add_argument('--fs-gscv-verbose', type=int, default=0, help="gscv verbosity")
parser.add_argument('--tr-splits', type=int, default=10, help='num tr splits')
parser.add_argument('--tr-cv-size', type=float, default=0.3, help="tr cv size")
parser.add_argument('--tr-gscv-splits', type=int, default=50, help='num tr gscv splits')
parser.add_argument('--tr-gscv-size', type=int, default=0.3, help='tr gscv size')
parser.add_argument('--tr-gscv-jobs', type=int, default=-1, help="tr gscv parallel jobs")
parser.add_argument('--tr-gscv-verbose', type=int, default=0, help="tr gscv verbosity")
parser.add_argument('--tr-rfecv-splits', type=int, default=32, help='num tr rfecv splits')
parser.add_argument('--tr-rfecv-size', type=int, default=0.3, help='rfecv cv size')
parser.add_argument('--tr-rfecv-jobs', type=int, default=-1, help="num tr rfecv parallel jobs")
parser.add_argument('--tr-rfecv-step', type=float, default=1, help="tr rfecv step")
parser.add_argument('--tr-rfecv-verbose', type=int, default=0, help="tr rfecv verbosity")
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--eset-tr', type=str, help="R eset for fs/tr")
parser.add_argument('--eset-te', type=str, help="R eset for te")
args = parser.parse_args()

def pipeline_one_eset(eset, fs_meth, tr_meth):
    X = np.array(base.t(biobase.exprs(eset)))
    y = np.array(r_filter_eset_relapse_labels(eset), dtype=int)
    results = []
    split_count = 0
    sss = StratifiedShuffleSplit(n_splits=args.tr_splits, test_size=args.tr_cv_size)
    for tr_idxs, te_idxs in sss.split(X, y):
        print('TR:', '%3s' % tr_idxs.size, ' TE:', '%3s' % te_idxs.size)
        X_tr, y_tr, X_te, y_te = X[tr_idxs], y[tr_idxs], X[te_idxs], y[te_idxs]
        split_results = tr_meth(X_tr, y_tr, X_te, y_te, eset, fs_meth(X_tr, y_tr, eset))
        results.append(split_results)
        split_count += 1
        # print(
        #     'TR split:', '%3s' % split_count,
        #     ' ROC AUC (Train):', '%.6f' % split_results['roc_auc_tr'],
        #     ' ROC AUC (Test):', '%.6f' % split_results['roc_auc_te'],
        # )
    # end for
    # print (
    #     'Mean ROC AUC (Train):', '%.6f' % roc_auc_tr_mean,
    #     ' Mean ROC AUC (Test):', '%.6f' % roc_auc,
    # )
    return(results)
# end pipeline

def pipeline_one_tr_multi_te(eset_tr, eset_tes, fs_meth, tr_meth):
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    fs_data = fs_meth(X_tr, y_tr, eset_tr)
    results = []
    for eset_te in eset_tes:
        X_te = np.array(base.t(biobase.exprs(eset_te)))
        y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
        eset_results = tr_meth(X_tr, y_tr, X_te, y_te, eset_tr, fs_data)
        results.append(eset_results)
    return(results)
# end pipeline

def pipeline_one_tr_one_te(eset_tr, eset_te, fs_meth, tr_meth):
    X_tr = np.array(base.t(biobase.exprs(eset_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_tr), dtype=int)
    X_te = np.array(base.t(biobase.exprs(eset_te)))
    y_te = np.array(r_filter_eset_relapse_labels(eset_te), dtype=int)
    results = tr_meth(X_tr, y_tr, X_te, y_te, eset_tr, fs_meth(X_tr, y_tr, eset_tr))
    return(results)
# end pipeline

def tr_meth_1(X_tr, y_tr, X_te, y_te, eset_tr, fs_data):
    tr_gscv_clf = GridSearchCV(
        Pipeline([
            ('slr', StandardScaler()),
            ('svc', LinearSVC(class_weight='balanced')),
        ]),
        param_grid=[
            # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
            { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        ],
        cv=StratifiedShuffleSplit(n_splits=args.tr_gscv_splits, test_size=args.tr_gscv_size),
        scoring='roc_auc', return_train_score=False, n_jobs=args.tr_gscv_jobs,
        verbose=args.tr_gscv_verbose
    )
    feature_idxs = fs_data['feature_idxs']
    feature_names = fs_data['feature_names']
    tr_data = {
        'fs_data': fs_data,
        'nf_split_data': [],
    }
    nf_split_count = 0
    for num_features in range(1, feature_idxs.size + 1):
        top_feature_idxs = feature_idxs[:num_features]
        top_feature_names = feature_names[:num_features]
        y_score = tr_gscv_clf.fit(X_tr[:,top_feature_idxs], y_tr).decision_function(X_te[:,top_feature_idxs])
        fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
        roc_auc = roc_auc_score(y_te, y_score)
        coefs = np.square(tr_gscv_clf.best_estimator_.named_steps['svc'].coef_[0])
        tr_data['nf_split_data'].append({
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
            'Split:', '%3s' % nf_split_count,
            ' ROC AUC (Train):', '%.4f' % tr_gscv_clf.best_score_,
            ' ROC AUC (Test):', '%.4f' % roc_auc,
        )
    # end for
    # tr_data = sorted(tr_data, key=lambda k: k['roc_auc_te']).pop()
    # print('Num Features:', tr_data['feature_idxs'].size)
    # for rank, feature, symbol in sorted(
    #     zip(
    #         tr_data['coefs'],
    #         tr_data['feature_names'],
    #         r_get_gene_symbols(eset_tr, robjects.IntVector(tr_data['feature_idxs'] + 1)),
    #     ),
    #     reverse=True
    # ): print(feature, "\t", symbol, "\t", rank)
    return(tr_data)
# end tr_meth_1

def tr_meth_2(X_tr, y_tr, X_te, y_te, eset_tr, fs_data):
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
            # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
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
    tr_data = {
        'gscv_clf': tr_gscv_clf,
        'fs_data': fs_data,
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
    ): print(feature, "\t", symbol, "\t", rank)
    return(tr_data)
# end tr_meth_2

def fs_limma_svm(X_tr, y_tr, eset_tr):
    fs_gscv_clf = GridSearchCV(
        Pipeline([
            ('slr', StandardScaler()),
            ('svc', LinearSVC(class_weight='balanced')),
        ]),
        param_grid=[
            # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
            { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        ],
        cv=StratifiedShuffleSplit(n_splits=args.fs_gscv_splits, test_size=args.fs_gscv_size),
        scoring='roc_auc', return_train_score=False, n_jobs=args.fs_gscv_jobs,
        verbose=args.fs_gscv_verbose
    )
    fs_data = {
        'split_data': [],
    }
    fs_split_count = 0
    low_fs_count = 0
    print_fs_header = True
    while fs_split_count < args.fs_splits:
        fs_idxs, cv_idxs = train_test_split(np.arange(y_tr.size), test_size=args.fs_cv_size, stratify=y_tr)
        if print_fs_header:
            print('FS:', '%3s' % fs_idxs.size, ' CV:', '%3s' % cv_idxs.size)
            print_fs_header = False
        feature_idxs = np.array(
            r_get_dfx_features(
                r_filter_eset(eset_tr, robjects.NULL, robjects.IntVector(fs_idxs + 1)),
                True,
                args.fs_dfx_pval,
                args.fs_dfx_lfc,
                args.fs_dfx_max,
            )
        ) - 1
        if feature_idxs.size < args.fs_dfx_min:
            low_fs_count += 1
            continue
        X_fs, y_fs, X_cv, y_cv = X_tr[fs_idxs], y_tr[fs_idxs], X_tr[cv_idxs], y_tr[cv_idxs]
        y_score = fs_gscv_clf.fit(X_fs[:,feature_idxs], y_fs).decision_function(X_cv[:,feature_idxs])
        fpr, tpr, thres = roc_curve(y_cv, y_score, pos_label=1)
        roc_auc = roc_auc_score(y_cv, y_score)
        fs_data['split_data'].append({
            'feature_idxs': feature_idxs,
            'fprs': fpr,
            'tprs': tpr,
            'thres': thres,
            'coefs': np.square(fs_gscv_clf.best_estimator_.named_steps['svc'].coef_[0]),
            'y_scores': y_score,
            'y_tests': y_cv,
            'roc_auc': roc_auc,
        })
        fs_split_count += 1
        print(
            'Split:', '%3s' % fs_split_count, ' Fails:', '%3s' % low_fs_count,
            ' FS:', '%3s' % feature_idxs.size, ' ROC AUC:', '%.4f' % roc_auc,
        )
    # end while
    feature_idxs = []
    for fs_split in fs_data['split_data']: feature_idxs.extend(fs_split['feature_idxs'])
    feature_idxs = sorted(list(set(feature_idxs)))
    feature_names = np.array(biobase.featureNames(eset_tr))
    feature_names = feature_names[feature_idxs]
    # print(*natsorted(feature_names), sep="\n")
    feature_mx_idx = {}
    for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
    coef_mx = np.zeros((len(feature_idxs), args.fs_splits), dtype=float)
    for split_idx in range(len(fs_data['split_data'])):
        split_data = fs_data['split_data'][split_idx]
        for idx in range(len(split_data['feature_idxs'])):
            coef_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                split_data['coefs'][idx]
    fs_data['feature_mean_coefs'] = []
    for idx in range(len(feature_idxs)):
        fs_data['feature_mean_coefs'].append(np.mean(coef_mx[idx]))
        # print(
        #     feature_names[idx], "\t",
        #     fs_data['feature_mean_coefs'][idx], "\t",
        #     coef_mx[idx]
        # )
    roc_auc_mx = np.zeros((len(feature_idxs), args.fs_splits), dtype=float)
    for split_idx in range(len(fs_data['split_data'])):
        split_data = fs_data['split_data'][split_idx]
        for idx in range(len(split_data['feature_idxs'])):
            roc_auc_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                split_data['roc_auc']
    fs_data['feature_mean_roc_aucs'] = []
    for idx in range(len(feature_idxs)):
        fs_data['feature_mean_roc_aucs'].append(np.mean(roc_auc_mx[idx]))
        # print(
        #     feature_names[idx], "\t",
        #     fs_data['feature_mean_roc_aucs'][idx], "\t",
        #     roc_auc_mx[idx]
        # )
    fs_data['feature_rank_data'] = sorted(
        zip(
            fs_data['feature_' + args.fs_rank_meth],
            feature_idxs,
            feature_names,
        ),
        reverse=True
    )
    fs_top_cutoff = min(args.fs_top_cutoff, len(feature_idxs))
    print('Num Features:', fs_top_cutoff, '/', len(feature_idxs))
    top_rank_data = fs_data['feature_rank_data'][:fs_top_cutoff]
    fs_data['feature_idxs'] = np.array([x for _, x, _ in top_rank_data], dtype=int)
    fs_data['feature_names'] = np.array([x for _, _, x in top_rank_data], dtype=str)
    return(fs_data)
# end fs_limma_svm

if (args.run == 1):
    eset_tr_name = 'eset_gex_gse31210'
    base.load("data/" + eset_tr_name + ".Rda")
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    results = pipeline_one_eset(eset_tr, fs_limma_svm, tr_meth_1)
    # plot roc curves
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train+Test\nROC Curves Using Limma+SVM Feature Selection (20 Top Ranked Features)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for idx, tr_split in enumerate(results):
        nf_split = tr_split['nf_split_data'][19]
        tprs.append(interp(mean_fpr, nf_split['fprs'], nf_split['tprs']))
        tprs[-1][0] = 0.0
        roc_aucs.append(nf_split['roc_auc_te'])
        plt.plot(
            nf_split['fprs'], nf_split['tprs'], lw=1, alpha=0.3,
            label='ROC split %d (AUC = %0.4f)' % (idx + 1, nf_split['roc_auc_te']),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=2, alpha=0.8,
        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=0.2,
        label=r'$\pm$ 1 std. dev.'
    )
    plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=0.8, label='Chance')
    plt.legend(loc='lower right')
    plt.grid('off')
    # plot num top ranked features selected vs mean roc auc
    roc_aucs_tr, roc_aucs_te = [], []
    for tr_split in results:
        for nf_idx, nf_split in enumerate(tr_split['nf_split_data']):
            if nf_idx < len(roc_aucs_tr):
                roc_aucs_tr[nf_idx].append(nf_split['roc_auc_tr'])
                roc_aucs_te[nf_idx].append(nf_split['roc_auc_te'])
            else:
                roc_aucs_tr.append([nf_split['roc_auc_tr']])
                roc_aucs_te.append([nf_split['roc_auc_te']])
    mean_roc_aucs_tr, mean_roc_aucs_te = [], []
    std_roc_aucs_tr, std_roc_aucs_te = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        mean_roc_aucs_te.append(np.mean(roc_aucs_te[nf_idx]))
        std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
        std_roc_aucs_te.append(np.std(roc_aucs_te[nf_idx]))
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train+Test\nEffect of Number Top Ranked Features Selected on ROC AUC')
    plt.xlabel("Number of top-ranked features selected")
    plt.ylabel("ROC AUC")
    plt.xlim([0.5, len(mean_roc_aucs_tr) + 0.5])
    plt_fig2_x_axis = range(1, len(mean_roc_aucs_tr) + 1)
    plt.xticks(plt_fig2_x_axis)
    plt.plot(
        plt_fig2_x_axis, mean_roc_aucs_tr,
        lw=2, label='Mean ROC AUC (Train)',
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
        lw=2, label='Mean ROC AUC (Test)',
        # label=r'Test Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (roc_auc_te_mean, roc_auc_te_std),
    )
    plt.fill_between(
        plt_fig2_x_axis,
        [m - s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
        [m + s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
        color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
    )
    plt.legend(loc='lower right')
    # show final selected feature information
    feature_idxs = []
    for split in results: feature_idxs.extend(split['nf_split_data'][19]['feature_idxs'])
    feature_idxs = sorted(list(set(feature_idxs)))
    feature_names = np.array(biobase.featureNames(eset_tr))
    feature_names = feature_names[feature_idxs]
    # print(*natsorted(feature_names), sep="\n")
    feature_mx_idx = {}
    for idx, feature_idx in enumerate(feature_idxs): feature_mx_idx[feature_idx] = idx
    coef_mx = np.zeros((len(feature_idxs), len(results)), dtype=float)
    for split_idx in range(len(results)):
        split_data = results[split_idx]['nf_split_data'][19]
        for idx in range(len(split_data['feature_idxs'])):
            coef_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
                split_data['coefs'][idx]
    feature_mean_coefs = []
    for idx in range(len(feature_idxs)):
        feature_mean_coefs.append(np.mean(coef_mx[idx]))
        # print(
        #     feature_names[idx], "\t",
        #     feature_mean_coefs[idx], "\t",
        #     coef_mx[idx]
        # )
    for rank, feature, symbol in sorted(
        zip(
            feature_mean_coefs,
            feature_names,
            r_get_gene_symbols(eset_tr, robjects.IntVector(np.array(feature_idxs) + 1)),
        ),
        reverse=True

    ): print(feature, "\t", symbol, "\t", rank)
elif args.run == 2:
    eset_tr_name = 'eset_gex_gse31210'
    base.load("data/" + eset_tr_name + ".Rda")
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    results = pipeline_one_eset(eset_tr, fs_limma_svm, tr_meth_2)
    # plot roc curves
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train+Test\nROC Curves Using Limma+SVM+RFECV Feature Selection')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for idx, tr_split in enumerate(results):
        tprs.append(interp(mean_fpr, tr_split['fprs'], tr_split['tprs']))
        tprs[-1][0] = 0.0
        roc_aucs.append(tr_split['roc_auc_te'])
        plt.plot(
            tr_split['fprs'], tr_split['tprs'], lw=1, alpha=0.3,
            label='ROC split %d (AUC = %0.4f)' % (idx + 1, tr_split['roc_auc_te']),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=2, alpha=0.8,
        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=0.2,
        label=r'$\pm$ 1 std. dev.'
    )
    plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=0.8, label='Chance')
    plt.legend(loc='lower right')
    plt.grid('off')
    # plot num features selected vs train roc auc
    roc_aucs_tr = []
    for tr_split in results:
        for nf_idx, roc_auc_tr in enumerate(tr_split['gscv_clf'].best_estimator_.named_steps['rfe'].grid_scores_):
            if nf_idx < len(roc_aucs_tr):
                roc_aucs_tr[nf_idx].append(roc_auc_tr)
            else:
                roc_aucs_tr.append([roc_auc_tr])
    mean_roc_aucs_tr, std_roc_aucs_tr = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train+Test\nEffect of Number of RFECV Features Selected on Train ROC AUC')
    plt.xlabel('Number of features selected')
    plt.ylabel('ROC AUC')
    max_features = len(results[0]['gscv_clf'].best_estimator_.named_steps['rfe'].grid_scores_)
    plt_fig2_x_axis = range(1, max_features + 1)
    plt.xlim([0.5, max_features + 0.5])
    plt.plot(
        plt_fig2_x_axis,
        mean_roc_aucs_tr,
        lw=2, label='Mean ROC AUC (Train)',
    )
    plt.fill_between(
        plt_fig2_x_axis,
        [m - s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        [m + s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
elif args.run == 3:
    eset_tr_name = 'eset_gex_gse31210'
    base.load("data/" + eset_tr_name + ".Rda")
    eset_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_name])
    eset_te_names = [
        "eset_gex_gse8894",
        "eset_gex_gse30219",
        "eset_gex_gse37745",
        "eset_gex_gse50081",
    ]
    te_colors = [
        "red",
        "blue",
        "green",
        "magenta",
    ]
    eset_tes = []
    for eset_te_name in eset_te_names:
        base.load("data/" + eset_te_name + ".Rda")
        eset_tes.append(r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_name]))
    results = pipeline_one_tr_multi_te(eset_tr, eset_tes, fs_limma_svm, tr_meth_1)
    # plot roc curves
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train Vs LUAD Test Datasets\nROC Curves Using Limma+SVM Feature Selection (Best Scoring Num Features)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    tprs, roc_aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    for idx, split in enumerate(results):
        nf_split = sorted(split['nf_split_data'], key=lambda k: k['roc_auc_te']).pop()
        tprs.append(interp(mean_fpr, nf_split['fprs'], nf_split['tprs']))
        tprs[-1][0] = 0.0
        roc_aucs.append(nf_split['roc_auc_te'])
        eset_te_name = eset_te_names[idx].replace('eset_gex_', '').upper()
        plt.plot(
            nf_split['fprs'], nf_split['tprs'], lw=2, alpha=0.5,
            label='%s ROC (AUC = %0.4f, Features = %d)' % (eset_te_name, nf_split['roc_auc_te'], len(nf_split['feature_idxs'])),
        )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='darkblue', lw=2, alpha=0.8,
        label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=0.2,
        label=r'$\pm$ 1 std. dev.'
    )
    plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=0.8, label='Chance')
    plt.legend(loc='lower right')
    plt.grid('off')
    # plot num top ranked features selected vs mean roc auc
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    plt.title('GSE31210 Train Vs LUAD Test Datasets\nEffect of Number Top Ranked Features Selected on ROC AUC')
    plt.xlabel("Number of top-ranked features selected")
    plt.ylabel("ROC AUC")
    max_features = len(results[0]['nf_split_data'])
    plt.xlim([0.5, max_features + 0.5])
    plt_fig2_x_axis = range(1, max_features + 1)
    plt.xticks(plt_fig2_x_axis)
    plt.plot(
        plt_fig2_x_axis, [s['roc_auc_tr'] for s in results[0]['nf_split_data']],
        lw=2, label='ROC AUC (Train) GSE31210',
    )
    for idx, split in enumerate(results):
        eset_te_name = eset_te_names[idx].replace('eset_gex_', '').upper()
        plt.plot(
            plt_fig2_x_axis, [s['roc_auc_te'] for s in split['nf_split_data']],
            lw=2, color=te_colors[idx], label='%s ROC AUC (Test)' % eset_te_name,
        )
    plt.legend(loc='lower right')
    # roc_aucs_tr, roc_aucs_te = [], []
    # for tr_split in results:
    #     for nf_idx, nf_split in enumerate(tr_split['nf_split_data']):
    #         if nf_idx < len(roc_aucs_tr):
    #             roc_aucs_tr[nf_idx].append(nf_split['roc_auc_tr'])
    #             roc_aucs_te[nf_idx].append(nf_split['roc_auc_te'])
    #         else:
    #             roc_aucs_tr.append([nf_split['roc_auc_tr']])
    #             roc_aucs_te.append([nf_split['roc_auc_te']])
    # mean_roc_aucs_tr, mean_roc_aucs_te = [], []
    # std_roc_aucs_tr, std_roc_aucs_te = [], []
    # for nf_idx in range(len(roc_aucs_tr)):
    #     mean_roc_aucs_tr.append(np.mean(roc_aucs_tr[nf_idx]))
    #     mean_roc_aucs_te.append(np.mean(roc_aucs_te[nf_idx]))
    #     std_roc_aucs_tr.append(np.std(roc_aucs_tr[nf_idx]))
    #     std_roc_aucs_te.append(np.std(roc_aucs_te[nf_idx]))
    # plt.fill_between(
    #     plt_fig2_x_axis,
    #     [m - s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
    #     [m + s for m, s in zip(mean_roc_aucs_tr, std_roc_aucs_tr)],
    #     color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    # )
    # plt.fill_between(
    #     plt_fig2_x_axis,
    #     [m - s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
    #     [m + s for m, s in zip(mean_roc_aucs_te, std_roc_aucs_te)],
    #     color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
    # )

# if args.eset_te:
#     base.load("data/" + args.eset_te + ".Rda")
#     eset_te = r_filter_eset_ctrl_probesets(robjects.globalenv[args.eset_te])

# eset_tr_strs = [
#     "eset_gex_gse31210_gse8894_gse30219_gse37745",
#     "eset_gex_gse31210_gse8894_gse30219_gse50081",
#     "eset_gex_gse31210_gse8894_gse37745_gse50081",
#     "eset_gex_gse31210_gse30219_gse37745_gse50081",
#     "eset_gex_gse8894_gse30219_gse37745_gse50081"
# ]
# eset_te_strs = [
#     "eset_gex_gse50081",
#     "eset_gex_gse37745",
#     "eset_gex_gse30219",
#     "eset_gex_gse8894",
#     "eset_gex_gse31210"
# ]

# pipeline 3
# eset_tr_str = "eset_gex_gse31210_tr_qnorm"
# eset_te_strs = [
#     "eset_gex_gse8894_te_qnorm",
#     "eset_gex_gse30219_te_qnorm",
#     "eset_gex_gse37745_te_qnorm",
#     "eset_gex_gse50081_te_qnorm",
# ]
# colors = [
#     'blue',
#     'green',
#     'red',
#     'magenta',
# ]
# base.load("data/" + eset_tr_str + ".Rda")
# eset_gex_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[eset_tr_str])
# plt_attrs = {
#     'label_tr': eset_tr_str,
#     'label_te': [],
#     'color': [],
# }
# eset_gex_te_s = []
# for idx, eset_te_str in enumerate(eset_te_strs):
#     base.load("data/" + eset_te_str + ".Rda")
#     eset_gex_te_s.append(r_filter_eset_ctrl_probesets(robjects.globalenv[eset_te_str]))
#     plt_attrs['label_te'].append(eset_te_str)
#     plt_attrs['color'].append(colors[idx])
# pipeline_3(eset_gex_tr, eset_gex_te_s, fs_meth_1, tr_meth_1, plt_attrs)

plt.show()
