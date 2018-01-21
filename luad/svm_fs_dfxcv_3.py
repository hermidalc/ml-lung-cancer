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
from sklearn.metrics import roc_curve, roc_auc_score
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
parser.add_argument('--fs-splits', type=int, default=100, help='num fs splits')
parser.add_argument('--tr-splits', type=int, default=100, help='num tr splits')
parser.add_argument('--fs-dfx-size', type=int, default=.5, help='fs dfx size')
parser.add_argument('--fs-cv-size', type=float, default=.1, help="fs cv size")
parser.add_argument('--tr-cv-size', type=float, default=.3, help="tr cv size")
parser.add_argument('--min-dfx-fs', type=int, default=10, help='min num dfx features to select')
parser.add_argument('--max-dfx-fs', type=int, default=100, help='min num dfx features to select')
parser.add_argument('--min-dfx-pval', type=float, default=.05, help="min dfx adj p value")
parser.add_argument('--min-dfx-lfc', type=float, default=1, help="min dfx logfc")
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--gscv-splits', type=int, default=10, help='num gridsearchcv splits')
parser.add_argument('--gscv-jobs', type=int, default=-1, help="num gridsearchcv parallel jobs")
parser.add_argument('--gscv-verbose', type=int, default=0, help="gridsearchcv verbosity")
parser.add_argument('--rfecv-splits', type=int, default=10, help='num rfecv splits')
parser.add_argument('--rfecv-jobs', type=int, default=-1, help="num rfecv parallel jobs")
parser.add_argument('--rfecv-step', type=float, default=1, help="rfecv step")
parser.add_argument('--rfecv-verbose', type=int, default=0, help="rfecv verbosity")
parser.add_argument('--eset-fs', type=str, help="R eset for fs")
parser.add_argument('--eset-tr', type=str, help="R eset for tr")
args = parser.parse_args()
fs_data = {
    'feature_idxs_all': [],
    'split_data': [],
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
    cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=0.2),
    scoring='roc_auc', return_train_score=False, n_jobs=args.gscv_jobs,
    verbose=args.gscv_verbose
)
base.load("data/" + args.eset_fs + ".Rda")
eset_gex = robjects.globalenv[args.eset_fs]
eset_gex = r_filter_eset_ctrl_probesets(eset_gex)
X = np.array(base.t(biobase.exprs(eset_gex)))
y = np.array(r_filter_eset_relapse_labels(eset_gex))
feature_names = np.array(biobase.featureNames(eset_gex))
split_count = 0
low_fs_count = 0
print_header = True
while split_count < args.fs_splits:
    tr_fs_idxs, cv_idxs = train_test_split(np.arange(y.size), test_size=args.fs_cv_size, stratify=y)
    tr_idxs, fs_idxs = train_test_split(tr_fs_idxs, test_size=args.fs_dfx_size, stratify=y[tr_fs_idxs])
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
    fs_data['split_data'].append({
        'feature_idxs': feature_idxs,
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(gscv_fs_clf.best_estimator_.named_steps['svc'].coef_[0]),
        'y_scores': y_score,
        'y_tests': y[cv_idxs],
        'roc_auc_score': roc_auc,
    })
    split_count += 1
    print('FS splits:', '%3s' % split_count, 'Fails:', '%3s' % low_fs_count, 'ROC AUC:', '%.4f' % roc_auc)
# end while
feature_idxs_fs = sorted(list(set(fs_data['feature_idxs_all'])))
feature_names_fs = feature_names[feature_idxs_fs]
print('Num Features:', len(feature_idxs_fs))
# print(*natsorted(feature_names_fs), sep="\n")
tr_data = {
    'feature_names': feature_names_fs,
    'y_scores_all': [],
    'y_tests_all': [],
    'split_data': [],
}
gscv_rfecv_tr_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('rfe',
            RFECV(
                LinearSVC(class_weight='balanced'), step=args.rfecv_step,
                cv=StratifiedShuffleSplit(n_splits=args.rfecv_splits, test_size=0.2),
                scoring='roc_auc', n_jobs=args.rfecv_jobs, verbose=args.rfecv_verbose
            )
        ),
        ('svc', LinearSVC(class_weight='balanced')),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=0.2),
    scoring='roc_auc', return_train_score=False, verbose=args.gscv_verbose
)
if args.eset_tr:
    base.load("data/" + args.eset_tr + ".Rda")
    eset_gex = robjects.globalenv[args.eset_tr]
    X = np.array(base.t(biobase.exprs(eset_gex)))
    y = np.array(r_filter_eset_relapse_labels(eset_gex))
X = X[:, feature_idxs_fs]
split_count = 0
print_header = True
while split_count < args.tr_splits:
    tr_idxs, cv_idxs = train_test_split(np.arange(y.size), test_size=args.tr_cv_size, stratify=y)
    if print_header:
        print('TR:', tr_idxs.size, 'CV:', cv_idxs.size)
        print_header = False
    y_score = gscv_rfecv_tr_clf.fit(X[tr_idxs], y[tr_idxs]).decision_function(X[cv_idxs])
    fpr, tpr, thres = roc_curve(y[cv_idxs], y_score, pos_label=1)
    roc_auc = roc_auc_score(y[cv_idxs], y_score)
    tr_data['y_scores_all'].extend(y_score.tolist())
    tr_data['y_tests_all'].extend(y[cv_idxs].tolist())
    rfecv_feature_idxs = gscv_rfecv_tr_clf.best_estimator_.named_steps['rfe'].get_support(indices=True)
    tr_data['split_data'].append({
        'clf': gscv_rfecv_tr_clf,
        'feature_ranks': sorted(zip(
            gscv_rfecv_tr_clf.best_estimator_.named_steps['rfe'].ranking_[rfecv_feature_idxs],
            feature_names_fs[rfecv_feature_idxs],
            r_get_gene_symbols(eset_gex, robjects.IntVector(rfecv_feature_idxs + 1))
        )),
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'y_scores': y_score,
        'y_tests': y[cv_idxs],
        'roc_auc_score': roc_auc,
    })
    split_count += 1
    print('TR splits:', '%3s' % split_count, 'ROC AUC:', '%.4f' % roc_auc)
best_cv_split = sorted(cv_data['split_data'], key=lambda k: k['roc_auc_score']).pop()
print(best_cv_split['clf'].best_score_)
for rank, feature, symbol in best_cv_split['feature_ranks']:
    print(feature, "\t", symbol, "\t", rank)
# save model
# joblib.dump(gscv_rfecv_tr_clf, 'data/svm_fs_dfxcv_2.pkl')
# save data
fs_data_fh = open('data/fs_data.pkl', 'wb')
tr_data_fh = open('data/tr_data.pkl', 'wb')
pickle.dump(fs_data, fs_data_fh, pickle.HIGHEST_PROTOCOL)
pickle.dump(tr_data, tr_data_fh, pickle.HIGHEST_PROTOCOL)
fs_data_fh.close()
tr_data_fh.close()
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(tr_data['y_tests_all'], tr_data['y_scores_all'], pos_label=1)
roc_auc = roc_auc_score(tr_data['y_tests_all'], tr_data['y_scores_all'])
print ('Overall ROC AUC:', '%.6f' % roc_auc)
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
