#!/usr/bin/env python

import argparse, math, statistics, pickle
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
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.externals import joblib
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
parser.add_argument('--fs-folds', type=int, default=100, help='num fs folds')
parser.add_argument('--cv-folds', type=int, default=100, help='num cv folds')
parser.add_argument('--cv-size', type=float, default=0.20, help="cv size")
parser.add_argument('--dfx-fs-size', type=int, default=0.5, help='num dfx fs size')
parser.add_argument('--min-dfx-fs', type=int, default=10, help='min num dfx features to select')
parser.add_argument('--max-dfx-fs', type=int, default=100, help='min num dfx features to select')
parser.add_argument('--min-dfx-pval', type=float, default=0.05, help="min dfx adj p value")
parser.add_argument('--min-dfx-lfc', type=float, default=1, help="min dfx logfc")
parser.add_argument('--top-fs', type=int, default=20, help='num top scoring features to select')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--fs-rank-method', type=str, default='mean_coefs', help="mean_coefs or mean_roc_auc_scores")
parser.add_argument('--gscv-folds', type=int, default=10, help='num gridsearchcv folds')
parser.add_argument('--gscv-jobs', type=int, default=-1, help="num gridsearchcv parallel jobs")
parser.add_argument('--gscv-verbose', type=int, default=1, help="gridsearchcv verbosity")
parser.add_argument('--eset-src', type=str, help="R eset for building svm")
parser.add_argument('--eset-cv', type=str, help="R eset for cross validation")
args = parser.parse_args()
fs_data = {
    'feature_idxs_all': [],
    'fold_data': [],
}
gscv_fs_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC(class_weight='balanced')),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.gscv_folds, test_size=0.2),
    scoring='roc_auc', return_train_score=False, n_jobs=args.gscv_jobs,
    verbose=1
)
base.load("data/" + args.eset_src + ".Rda")
eset_gex = robjects.globalenv[args.eset_src]
eset_gex = r_filter_eset_ctrl_probesets(eset_gex)
X = np.array(base.t(biobase.exprs(eset_gex)))
y = np.array(r_filter_eset_relapse_labels(eset_gex))
feature_names = np.array(biobase.featureNames(eset_gex))
fold_count = 0
low_fs_count = 0
print_header = True
while fold_count < args.fs_folds:
    tr_fs_idxs, cv_idxs = train_test_split(np.arange(y.size), test_size=0.1, stratify=y)
    tr_idxs, fs_idxs = train_test_split(tr_fs_idxs, test_size=args.dfx_fs_size, stratify=y[tr_fs_idxs])
    if print_header:
        print('FS:', fs_idxs.size, 'TR:', tr_idxs.size, 'CV:', cv_idxs.size)
        print_header = False
    feature_idxs = np.array(
        r_get_dfx_features(
            r_filter_eset(eset_gex, robjects.NULL, robjects.IntVector(fs_idxs + 1)),
            True,
            args.min_dfx_pval,
            args.min_dfx_lfc,
            args.max_dfx_fs,
        )
    ) - 1
    if feature_idxs.size < args.min_dfx_fs:
        low_fs_count += 1
        continue
    y_score = gscv_fs_clf.fit(X[np.ix_(tr_idxs, feature_idxs)], y[tr_idxs]).decision_function(X[np.ix_(cv_idxs, feature_idxs)])
    fpr, tpr, thres = roc_curve(y[cv_idxs], y_score, pos_label=1)
    roc_auc = roc_auc_score(y[cv_idxs], y_score)
    fs_data['feature_idxs_all'].extend(feature_idxs.tolist())
    fs_data['fold_data'].append({
        'feature_idxs': feature_idxs,
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(gscv_fs_clf.best_estimator_.named_steps['svc'].coef_[0]),
        'y_scores': y_score,
        'y_tests': y[cv_idxs],
        'roc_auc_score': roc_auc,
    })
    fold_count += 1
    print('FS Folds:', fold_count, 'Fails:', low_fs_count, 'ROC AUC:', roc_auc)
feature_idxs_fs = sorted(list(set(fs_data['feature_idxs_all'])))
feature_names_fs = feature_names[feature_idxs_fs]
# print(*natsorted(feature_names_fs), sep="\n")
feature_mx_idx = {}
for idx, feature_idx in enumerate(feature_idxs_fs): feature_mx_idx[feature_idx] = idx
coef_mx = np.zeros((len(feature_idxs_fs), args.fs_folds), dtype="float64")
for fold_idx in range(len(fs_data['fold_data'])):
    fold_data = fs_data['fold_data'][fold_idx]
    for idx in range(len(fold_data['feature_idxs'])):
        coef_mx[feature_mx_idx[fold_data['feature_idxs'][idx]]][fold_idx] = \
            fold_data['coefs'][idx]
fs_data['feature_mean_coefs'] = []
for idx in range(len(feature_idxs_fs)):
    fs_data['feature_mean_coefs'].append(
        statistics.mean(coef_mx[idx])
    )
    # print(
    #     feature_names_fs[idx], "\t",
    #     fs_data['feature_mean_coefs'][idx], "\t",
    #     coef_mx[idx]
    # )
roc_auc_score_mx = np.zeros((len(feature_idxs_fs), args.fs_folds), dtype="float64")
for fold_idx in range(len(fs_data['fold_data'])):
    fold_data = fs_data['fold_data'][fold_idx]
    for idx in range(len(fold_data['feature_idxs'])):
        roc_auc_score_mx[feature_mx_idx[fold_data['feature_idxs'][idx]]][fold_idx] = \
            fold_data['roc_auc_score']
fs_data['feature_mean_roc_auc_scores'] = []
for idx in range(len(feature_idxs_fs)):
    fs_data['feature_mean_roc_auc_scores'].append(
        statistics.mean(roc_auc_score_mx[idx])
    )
    # print(
    #     feature_names_fs[idx], "\t",
    #     fs_data['feature_mean_roc_auc_scores'][idx], "\t",
    #     roc_auc_score_mx[idx]
    # )
feature_ranks = sorted(
    zip(
        fs_data['feature_' + args.fs_rank_method],
        feature_names_fs,
        r_get_gene_symbols(eset_gex, robjects.StrVector(feature_names_fs))
    ),
    reverse=True
)
print('Num Features:', args.top_fs, '/', len(feature_idxs_fs))
feature_ranks = feature_ranks[:args.top_fs]
feature_names_fs = [x for _, x, _ in feature_ranks]
cv_data = {
    'feature_names': feature_names_fs,
    'y_scores_all': [],
    'y_tests_all': [],
    'fold_data': [],
}
gscv_cv_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC(class_weight='balanced')),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.gscv_folds, test_size=0.2),
    scoring='roc_auc', return_train_score=False, n_jobs=args.gscv_jobs,
    verbose=args.gscv_verbose
)
if args.eset_cv:
    base.load("data/" + args.eset_cv + ".Rda")
    eset_gex = robjects.globalenv[args.eset_cv]
    X = np.array(base.t(biobase.exprs(eset_gex)))
    y = np.array(r_filter_eset_relapse_labels(eset_gex))
fold_count = 0
print_header = True
while fold_count < args.cv_folds:
    tr_idxs, cv_idxs = train_test_split(np.arange(y.size), test_size=args.cv_size, stratify=y)
    if print_header:
        print('TR:', tr_idxs.size, 'CV:', cv_idxs.size)
        print_header = False
    y_score = gscv_cv_clf.fit(X[np.ix_(tr_idxs, feature_idxs_fs)], y[tr_idxs]).decision_function(X[np.ix_(cv_idxs, feature_idxs_fs)])
    fpr, tpr, thres = roc_curve(y[cv_idxs], y_score, pos_label=1)
    roc_auc = roc_auc_score(y[cv_idxs], y_score)
    cv_data['y_scores_all'].extend(y_score.tolist())
    cv_data['y_tests_all'].extend(y[cv_idxs].tolist())
    cv_data['fold_data'].append({
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'y_scores': y_score,
        'y_tests': y[cv_idxs],
        'roc_auc_score': roc_auc,
    })
    fold_count += 1
    print('CV Folds:', fold_count, 'ROC AUC:', roc_auc)
for rank, feature, symbol in feature_ranks: print(feature, "\t", symbol, "\t", rank)
# save data
fs_data_fh = open('data/fs_data.pkl', 'wb')
cv_data_fh = open('data/cv_data.pkl', 'wb')
pickle.dump(fs_data, fs_data_fh, pickle.HIGHEST_PROTOCOL)
pickle.dump(cv_data, cv_data_fh, pickle.HIGHEST_PROTOCOL)
fs_data_fh.close()
cv_data_fh.close()
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(cv_data['y_tests_all'], cv_data['y_scores_all'], pos_label=1)
roc_auc = roc_auc_score(cv_data['y_tests_all'], cv_data['y_scores_all'])
plt.rcParams['font.size'] = 24
plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=.8, label='Chance')
plt.plot(fpr, tpr, color='darkblue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
