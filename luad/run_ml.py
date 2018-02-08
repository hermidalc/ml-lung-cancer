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
parser.add_argument('--fs-splits', type=int, default=100, help='num fs splits')
parser.add_argument('--fs-cv-size', type=float, default=0.3, help="fs cv size")
parser.add_argument('--fs-dfx-min', type=int, default=5, help='fs min num dfx features')
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

def pipeline_single_eset(eset_gex, fs_meth, tr_meth):
    X = np.array(base.t(biobase.exprs(eset_gex)))
    y = np.array(r_filter_eset_relapse_labels(eset_gex), dtype=int)
    tr_data = []
    tr_split_count = 0
    sss = StratifiedShuffleSplit(n_splits=args.tr_splits, test_size=args.tr_cv_size)
    for tr_idxs, te_idxs in sss.split(X, y):
        print('TR:', '%3s' % tr_idxs.size, ' TE:', '%3s' % te_idxs.size)
        X_tr, y_tr, X_te, y_te = X[tr_idxs], y[tr_idxs], X[te_idxs], y[te_idxs]
        tr_split_data = tr_meth(X_tr, y_tr, X_te, y_te, eset_gex, fs_meth(X_tr, y_tr, eset_gex))
        tr_data.append(tr_split_data)
        tr_split_count += 1
        print(
            'TR split:', '%3s' % tr_split_count,
            #' ROC AUC (Train):', '%.6f' % tr_split_data['roc_auc_tr'],
            #' ROC AUC (Test):', '%.6f' % tr_split_data['roc_auc_te'],
        )
    # end for
    roc_aucs_tr, roc_aucs_te = [], []
    for tr_split in tr_data:
        for nf_idx, nf_split in enumerate(tr_split['nf_split_data']):
            if nf_idx < len(roc_aucs_tr):
                roc_aucs_tr[nf_idx].append(nf_split['roc_auc_tr'])
                roc_aucs_te[nf_idx].append(nf_split['roc_auc_te'])
            else:
                roc_aucs_tr.append([nf_split['roc_auc_tr']])
                roc_aucs_te.append([nf_split['roc_auc_te']])
    roc_auc_means_tr, roc_auc_means_te = [], []
    roc_auc_stds_tr, roc_auc_stds_te = [], []
    for nf_idx in range(len(roc_aucs_tr)):
        roc_auc_means_tr.append(np.mean(roc_aucs_tr[nf_idx]))
        roc_auc_means_te.append(np.mean(roc_aucs_te[nf_idx]))
        roc_auc_stds_tr.append(np.std(roc_aucs_tr[nf_idx]))
        roc_auc_stds_te.append(np.std(roc_aucs_te[nf_idx]))
    # print (
    #     'Mean ROC AUC (Train):', '%.6f' % roc_auc_tr_mean,
    #     ' Mean ROC AUC (Test):', '%.6f' % roc_auc,
    # )
    # plot num top features select vs roc auc
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title('Effect of Number Top Features Selected on ROC AUC')
    plt.xlabel("Number of features")
    plt.ylabel("ROC AUC")
    plt_x_axis = range(1, len(roc_auc_means_tr) + 1)
    plt.xticks(plt_x_axis)
    plt.plot(
        plt_x_axis, roc_auc_means_tr,
        lw=2, label='Train Mean ROC AUC',
        # label=r'Train Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (roc_auc_tr_mean, roc_auc_tr_std),
    )
    plt.fill_between(
        plt_x_axis,
        [m - s for m, s in zip(roc_auc_means_tr, roc_auc_stds_tr)],
        [m + s for m, s in zip(roc_auc_means_tr, roc_auc_stds_tr)],
        color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )
    plt.plot(
        plt_x_axis, roc_auc_means_te,
        lw=2, label='Test Mean ROC AUC',
        # label=r'Test Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (roc_auc_te_mean, roc_auc_te_std),
    )
    plt.fill_between(
        plt_x_axis,
        [m - s for m, s in zip(roc_auc_means_te, roc_auc_stds_te)],
        [m + s for m, s in zip(roc_auc_means_te, roc_auc_stds_te)],
        color='grey', alpha=0.2, #label=r'$\pm$ 1 std. dev.'
    )
    plt.legend(loc='lower right')
    # y_tests, y_scores = [], []
    # for tr_split in tr_data:
    #     y_tests.extend(tr_split['y_tests'])
    #     y_scores.extend(tr_split['y_scores'])
    # fpr, tpr, thres = roc_curve(y_tests, y_scores, pos_label=1)
    # roc_auc = roc_auc_score(y_tests, y_scores)
    # plot roc auc
    # plt.figure(2)
    # plt.rcParams['font.size'] = 20
    # plt.plot([0,1], [0,1], color='red', lw=2, linestyle='--', alpha=.8, label='Chance')
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc)
    # plt.xlim([0,1.01])
    # plt.ylim([0,1.01])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
# end pipeline1

def tr_meth_1(X_tr, y_tr, X_te, y_te, eset_gex_tr, fs_data):
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
    #         r_get_gene_symbols(eset_gex_tr, robjects.IntVector(tr_data['feature_idxs'] + 1)),
    #     ),
    #     reverse=True
    # ): print(feature, "\t", symbol, "\t", rank)
    return(tr_data)
# end tr_meth_1

def tr_meth_2(X_tr, y_tr, X_te, y_te, eset_gex_tr, fs_data, plt_attrs):
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
    print('Num Features:', tr_gscv_clf.best_estimator_.named_steps['rfe'].n_features_)
    for rank, feature, symbol in sorted(
        zip(
            coefs,
            rfe_feature_names,
            r_get_gene_symbols(eset_gex_tr, robjects.IntVector(rfe_feature_idxs + 1)),
        ),
        reverse=True
    ): print(feature, "\t", symbol, "\t", rank)
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
    # plot num features selected vs train roc auc
    plt.figure(1)
    plt.rcParams['font.size'] = 20
    plt.title("Effect of Number of Features Selected on Training ROC AUC")
    plt.xlabel("Number of features")
    plt.ylabel("ROC AUC")
    plt.plot(
        range(1, len(tr_gscv_clf.best_estimator_.named_steps['rfe'].grid_scores_) + 1),
        tr_gscv_clf.best_estimator_.named_steps['rfe'].grid_scores_,
        color=plt_attrs['color'],
    )
    return(tr_data)
# end tr_meth_2

def fs_limma_svm(X_tr, y_tr, eset_gex_tr):
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
                r_filter_eset(eset_gex_tr, robjects.NULL, robjects.IntVector(fs_idxs + 1)),
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
    feature_names = np.array(biobase.featureNames(eset_gex_tr))
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

def pipeline_2(eset_gex_tr, eset_gex_te, fs_meth, tr_meth, plt_attrs):
    X_tr = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_gex_tr), dtype=int)
    X_te = np.array(base.t(biobase.exprs(eset_gex_te)))
    y_te = np.array(r_filter_eset_relapse_labels(eset_gex_te), dtype=int)
    feature_names = np.array(biobase.featureNames(eset_gex_tr))
    tr_data = tr_meth(X_tr, y_tr, X_te, y_te, eset_gex, fs_meth(X_tr, y_tr, eset_gex_tr), [], 0)
    fpr, tpr, thres = roc_curve(tr_data['y_tests'], tr_data['y_scores'], pos_label=1)
    roc_auc = roc_auc_score(tr_data['y_tests'], tr_data['y_scores'])
    # roc_auc_std = np.std(aucs)
    print ('ROC AUC:', '%.6f' % roc_auc)
    # plot roc auc
    plt.figure(2)
    plt.rcParams['font.size'] = 20
    plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=.8, label='Chance')
    plt.plot(fpr, tpr, color=plt_attrs['color'], lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.xlim([0,1.01])
    plt.ylim([0,1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
# end pipeline2

def pipeline_3(eset_gex_tr, eset_gex_te_s, fs_meth, tr_meth, plt_attrs):
    print(plt_attrs['label_tr'])
    X_tr = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_gex_tr), dtype=int)
    feature_names = np.array(biobase.featureNames(eset_gex_tr))
    fs_data = fs_meth(X_tr, y_tr, eset_gex_tr)
    for plt_idx, eset_gex_te in enumerate(eset_gex_te_s):
        print(plt_attrs['label_te'][plt_idx])
        X_te = np.array(base.t(biobase.exprs(eset_gex_te)))
        y_te = np.array(r_filter_eset_relapse_labels(eset_gex_te), dtype=int)
        tr_data = tr_meth(X_tr, y_tr, X_te, y_te, eset_gex_tr, fs_data, plt_attrs, plt_idx)
        fpr, tpr, thres = roc_curve(tr_data['y_tests'], tr_data['y_scores'], pos_label=1)
        roc_auc = roc_auc_score(tr_data['y_tests'], tr_data['y_scores'])
        print ('ROC AUC:', '%.6f' % roc_auc)
# end pipeline 3

# pipeline single eset
base.load("data/" + args.eset_tr + ".Rda")
eset_gex_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[args.eset_tr])
pipeline_single_eset(eset_gex_tr, fs_limma_svm, tr_meth_1)

# if args.eset_te:
#     base.load("data/" + args.eset_te + ".Rda")
#     eset_gex_te = r_filter_eset_ctrl_probesets(robjects.globalenv[args.eset_te])

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
