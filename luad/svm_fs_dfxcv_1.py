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
parser.add_argument('--fs-cv-size', type=float, default=0.2, help="fs cv size")
parser.add_argument('--fs-dfx-min', type=int, default=10, help='fs min num dfx features')
parser.add_argument('--fs-dfx-max', type=int, default=100, help='fs max num dfx features')
parser.add_argument('--fs-dfx-pval', type=float, default=0.01, help="min dfx adj p value")
parser.add_argument('--fs-dfx-lfc', type=float, default=0, help="min dfx logfc")
parser.add_argument('--fs-rank-meth', type=str, default='mean_coefs', help="mean_coefs or mean_roc_auc_scores")
parser.add_argument('--fs-top-cutoff', type=int, default=15, help='fs top ranked features cutoff')
parser.add_argument('--fs-gscv-splits', type=int, default=20, help='num fs gscv splits')
parser.add_argument('--fs-gscv-size', type=int, default=0.3, help='fs gscv cv size')
parser.add_argument('--fs-gscv-jobs', type=int, default=-1, help="num gscv parallel jobs")
parser.add_argument('--fs-gscv-verbose', type=int, default=0, help="gscv verbosity")
parser.add_argument('--tr-splits', type=int, default=100, help='num tr splits')
parser.add_argument('--tr-gscv-splits', type=int, default=20, help='num tr gscv splits')
parser.add_argument('--tr-gscv-size', type=int, default=0.3, help='tr gscv size')
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
fs_gscv_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC()),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.fs_gscv_splits, test_size=args.fs_gscv_size),
    scoring='roc_auc', return_train_score=False, n_jobs=args.fs_gscv_jobs,
    verbose=args.fs_gscv_verbose
)
base.load("data/" + args.eset_tr + ".Rda")
eset_gex_tr = r_filter_eset_ctrl_probesets(robjects.globalenv[args.eset_tr])
X_tr = np.array(base.t(biobase.exprs(eset_gex_tr)))
y_tr = np.array(r_filter_eset_relapse_labels(eset_gex_tr), dtype=int)
base.load("data/" + args.eset_te + ".Rda")
eset_gex_te = r_filter_eset_ctrl_probesets(robjects.globalenv[args.eset_te])
X_te = np.array(base.t(biobase.exprs(eset_gex_te)))
y_te = np.array(r_filter_eset_relapse_labels(eset_gex_te), dtype=int)
feature_names = np.array(biobase.featureNames(eset_gex_tr))
fs_data = {
    'feature_idxs': [],
    'split_data': [],
}
fs_split_count = 0
low_fs_count = 0
print_fs_header = True
while fs_split_count < args.fs_splits:
    y_tr_r_idxs = shuffle(np.where(y_tr == 1)[0])
    y_tr_n_idxs = shuffle(np.where(y_tr == 0)[0])
    if args.fs_cv_size >= 1:
        fs_num_r = args.fs_cv_size
    else:
        fs_num_r = math.ceil(y_tr_r_idxs.size * args.fs_cv_size)
    fs_num_n = y_tr_n_idxs.size - y_tr_r_idxs.size + fs_num_r
    tr_num = y_tr_r_idxs.size - (fs_num_r * 2)
    cv_num = y_tr_r_idxs.size - fs_num_r - tr_num
    fs_idxs = np.concatenate((y_tr_r_idxs[:fs_num_r], y_tr_n_idxs[:fs_num_n]))
    tr_idxs = np.concatenate((y_tr_r_idxs[fs_num_r:(fs_num_r + tr_num)], y_tr_n_idxs[fs_num_n:(fs_num_n + tr_num)]))
    cv_idxs = np.concatenate((y_tr_r_idxs[(fs_num_r + tr_num):], y_tr_n_idxs[(fs_num_n + tr_num):]))
    if print_fs_header:
        print('FS:', fs_num_r, '/', fs_num_n, ' TR:', tr_num, '/', tr_num, ' CV:', cv_num, '/', cv_num)
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
    X_ftr, y_ftr, X_cv, y_cv = X_tr[tr_idxs], y_tr[tr_idxs], X_tr[cv_idxs], y_tr[cv_idxs]
    y_score = fs_gscv_clf.fit(X_ftr[:,feature_idxs], y_ftr).decision_function(X_cv[:,feature_idxs])
    fpr, tpr, thres = roc_curve(y_cv, y_score, pos_label=1)
    roc_auc = roc_auc_score(y_cv, y_score)
    fs_data['feature_idxs'].extend(feature_idxs.tolist())
    fs_data['split_data'].append({
        'feature_idxs': feature_idxs,
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(fs_gscv_clf.best_estimator_.named_steps['svc'].coef_[0]),
        'y_scores': y_score,
        'y_tests': y_cv,
        'roc_auc_score': roc_auc,
    })
    fs_split_count += 1
    print(
        'Split:', '%3s' % fs_split_count, ' Fails:', '%3s' % low_fs_count,
        ' FS:', '%3s' % feature_idxs.size, ' ROC AUC:', '%.4f' % roc_auc,
    )
# end while
fs_feature_idxs = sorted(list(set(fs_data['feature_idxs'])))
fs_feature_names = feature_names[fs_feature_idxs]
# print(*natsorted(fs_feature_names), sep="\n")
feature_mx_idx = {}
for idx, feature_idx in enumerate(fs_feature_idxs): feature_mx_idx[feature_idx] = idx
coef_mx = np.zeros((len(fs_feature_idxs), args.fs_splits), dtype=float)
for split_idx in range(len(fs_data['split_data'])):
    split_data = fs_data['split_data'][split_idx]
    for idx in range(len(split_data['feature_idxs'])):
        coef_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
            split_data['coefs'][idx]
fs_data['feature_mean_coefs'] = []
for idx in range(len(fs_feature_idxs)):
    fs_data['feature_mean_coefs'].append(
        statistics.mean(coef_mx[idx])
    )
    # print(
    #     fs_feature_names[idx], "\t",
    #     fs_data['feature_mean_coefs'][idx], "\t",
    #     coef_mx[idx]
    # )
roc_auc_score_mx = np.zeros((len(fs_feature_idxs), args.fs_splits), dtype=float)
for split_idx in range(len(fs_data['split_data'])):
    split_data = fs_data['split_data'][split_idx]
    for idx in range(len(split_data['feature_idxs'])):
        roc_auc_score_mx[feature_mx_idx[split_data['feature_idxs'][idx]]][split_idx] = \
            split_data['roc_auc_score']
fs_data['feature_mean_roc_auc_scores'] = []
for idx in range(len(fs_feature_idxs)):
    fs_data['feature_mean_roc_auc_scores'].append(
        statistics.mean(roc_auc_score_mx[idx])
    )
    # print(
    #     fs_feature_names[idx], "\t",
    #     fs_data['feature_mean_roc_auc_scores'][idx], "\t",
    #     roc_auc_score_mx[idx]
    # )
feature_rank_data = sorted(
    zip(
        fs_data['feature_' + args.fs_rank_meth],
        fs_feature_idxs,
        fs_feature_names,
        r_get_gene_symbols(eset_gex_tr, robjects.StrVector(fs_feature_names)),
    ),
    reverse=True
)
fs_top_cutoff = min(args.fs_top_cutoff, len(fs_feature_idxs))
print('Num Features:', fs_top_cutoff, '/', len(fs_feature_idxs))
feature_rank_data = feature_rank_data[:fs_top_cutoff]
# for rank, idx, name, symbol in feature_rank_data: print(name, "\t", symbol, "\t", rank)
fs_feature_idxs = np.array([x for _, x, _, _ in feature_rank_data], dtype=int)
fs_feature_names = np.array([x for _, _, x, _ in feature_rank_data], dtype=str)
# train and test classifier
tr_gscv_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC()),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.tr_gscv_splits, test_size=args.tr_gscv_size),
    scoring='roc_auc', return_train_score=False, verbose=args.tr_gscv_verbose
)
tr_data = {
    'feature_idxs': fs_feature_idxs,
    'feature_names': fs_feature_names,
    'y_scores': [],
    'y_tests': [],
    'split_data': [],
}
tr_split_count = 0
while tr_split_count < args.tr_splits:
    y_tr_r_idxs = np.where(y_tr == 1)[0]
    y_tr_n_idxs = shuffle(np.where(y_tr == 0)[0])
    tr_idxs = np.concatenate((y_tr_r_idxs, y_tr_n_idxs[:y_tr_r_idxs.size]))
    X_trn, y_trn = X_tr[tr_idxs], y_tr[tr_idxs]
    y_score = tr_gscv_clf.fit(X_trn[:,fs_feature_idxs], y_trn).decision_function(X_te[:,fs_feature_idxs])
    fpr, tpr, thres = roc_curve(y_te, y_score, pos_label=1)
    roc_auc = roc_auc_score(y_te, y_score)
    tr_data['y_scores'].extend(y_score.tolist())
    tr_data['y_tests'].extend(y_te.tolist())
    tr_data['split_data'].append({
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(tr_gscv_clf.best_estimator_.named_steps['svc'].coef_[0]),
        'y_scores': y_score,
        'y_tests': y_te,
        'roc_auc_score': roc_auc,
    })
    tr_split_count += 1
    print('TR splits:', '%3s' % tr_split_count, ' ROC AUC:', '%.4f' % roc_auc)
# end while
for rank, idx, feature, symbol in feature_rank_data: print(feature, "\t", symbol, "\t", rank)
# save data
# fs_data_fh = open('data/fs_data.pkl', 'wb')
# tr_data_fh = open('data/tr_data.pkl', 'wb')
# pickle.dump(fs_data, fs_data_fh, pickle.HIGHEST_PROTOCOL)
# pickle.dump(tr_data, tr_data_fh, pickle.HIGHEST_PROTOCOL)
# fs_data_fh.close()
# tr_data_fh.close()
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(tr_data['y_tests'], tr_data['y_scores'], pos_label=1)
roc_auc = roc_auc_score(tr_data['y_tests'], tr_data['y_scores'])
print ('ROC AUC:', '%.6f' % roc_auc)
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
