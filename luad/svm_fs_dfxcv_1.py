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
parser.add_argument('--fs-splits', type=int, default=100, help='num feature selection splits')
parser.add_argument('--tr-splits', type=int, default=100, help='num training splits')
parser.add_argument('--tr-cv-size', type=float, default=.3, help="training cv size")
parser.add_argument('--fs-dfx-relapse', type=int, default=10, help='num dfx fs relapse samples')
parser.add_argument('--min-dfx-fs', type=int, default=10, help='min num dfx features to select')
parser.add_argument('--max-dfx-fs', type=int, default=100, help='min num dfx features to select')
parser.add_argument('--min-dfx-pval', type=float, default=.05, help="min dfx adj p value")
parser.add_argument('--min-dfx-lfc', type=float, default=1, help="min dfx logfc")
parser.add_argument('--top-fs', type=int, default=10, help='num top scoring features to select')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--fs-rank-method', type=str, default='mean_coefs', help="mean_coefs or mean_roc_auc_scores")
parser.add_argument('--gscv-splits', type=int, default=10, help='num gridsearchcv splits')
parser.add_argument('--gscv-jobs', type=int, default=-1, help="num gridsearchcv parallel jobs")
parser.add_argument('--gscv-verbose', type=int, default=0, help="gridsearchcv verbosity")
parser.add_argument('--eset-fs', type=str, help="R eset for fs")
parser.add_argument('--eset-tr', type=str, help="R eset for tr")
args = parser.parse_args()
fs_data = {
    'features_all': [],
    'split_data': [],
}
gscv_fs_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC()),
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
split_count = 0
low_fs_count = 0
print_header = True
while split_count < args.fs_splits:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    if args.fs_dfx_relapse >= 1:
        num_relapse_samples_fs = args.fs_dfx_relapse
    else
        num_relapse_samples_fs = math.ceil(len(relapse_samples) * args.fs_dfx_relapse)
    num_norelapse_samples_fs = len(norelapse_samples) - len(relapse_samples) + num_relapse_samples_fs
    samples_fs = relapse_samples[:num_relapse_samples_fs] + \
                 norelapse_samples[:num_norelapse_samples_fs]
    features = r_get_dfx_features(
        r_filter_eset(eset_gex, robjects.NULL, samples_fs),
        False,
        args.min_dfx_pval,
        args.min_dfx_lfc,
        args.max_dfx_fs,
    )
    if len(features) < args.min_dfx_fs:
        low_fs_count += 1
        continue
    num_samples_tr = len(relapse_samples) - (num_relapse_samples_fs * 2)
    samples_tr = relapse_samples[num_relapse_samples_fs:(num_relapse_samples_fs + num_samples_tr)] + \
                 norelapse_samples[num_norelapse_samples_fs:(num_norelapse_samples_fs + num_samples_tr)]
    num_samples_ts = len(relapse_samples) - num_relapse_samples_fs - num_samples_tr
    samples_ts = relapse_samples[(num_relapse_samples_fs + num_samples_tr):] + \
                 norelapse_samples[(num_norelapse_samples_fs + num_samples_tr):]
    if print_header:
        print(
            'FS:', num_relapse_samples_fs, '/', num_norelapse_samples_fs,
            'TR:', num_samples_tr, '/', num_samples_tr,
            'CV:', num_samples_ts, '/', num_samples_ts,
        )
        print_header = False
    eset_gex_tr = r_filter_eset(eset_gex, features, samples_tr)
    X_train = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    y_score = gscv_fs_clf.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_score)
    fs_data['features_all'].extend(np.array(features).tolist())
    fs_data['split_data'].append({
        'features': np.array(features).tolist(),
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(gscv_fs_clf.best_estimator_.named_steps['svc'].coef_[0]),
        'y_scores': y_score,
        'y_tests': y_test,
        'roc_auc_score': roc_auc,
    })
    split_count += 1
    print('FS splits:', '%3s' % split_count, 'Fails:', '%3s' % low_fs_count, 'ROC AUC:', '%.4f' % roc_auc)
# end while
# rank features
fs_data['features_uniq'] = natsorted(list(set(fs_data['features_all'])))
# print(*fs_data['features_uniq'], sep="\n")
feature_mx_idx = {}
for idx, feature in enumerate(fs_data['features_uniq']): feature_mx_idx[feature] = idx
coef_mx = np.zeros((len(fs_data['features_uniq']), args.fs_splits), dtype="float64")
for split_idx in range(len(fs_data['split_data'])):
    split_data = fs_data['split_data'][split_idx]
    for feature_idx in range(len(split_data['features'])):
        coef_mx[feature_mx_idx[split_data['features'][feature_idx]]][split_idx] = \
            split_data['coefs'][feature_idx]
fs_data['feature_mean_coefs'] = []
for feature_idx in range(len(fs_data['features_uniq'])):
    fs_data['feature_mean_coefs'].append(
        statistics.mean(coef_mx[feature_idx])
    )
    # print(
    #     fs_data['features_uniq'][feature_idx], "\t",
    #     fs_data['feature_mean_coefs'][feature_idx], "\t",
    #     coef_mx[feature_idx]
    # )
roc_auc_score_mx = np.zeros((len(fs_data['features_uniq']), args.fs_splits), dtype="float64")
for split_idx in range(len(fs_data['split_data'])):
    split_data = fs_data['split_data'][split_idx]
    for feature in split_data['features']:
        roc_auc_score_mx[feature_mx_idx[feature]][split_idx] = \
            split_data['roc_auc_score']
fs_data['feature_mean_roc_auc_scores'] = []
for feature_idx in range(len(fs_data['features_uniq'])):
    fs_data['feature_mean_roc_auc_scores'].append(
        statistics.mean(roc_auc_score_mx[feature_idx])
    )
    # print(
    #     fs_data['features_uniq'][feature_idx], "\t",
    #     fs_data['feature_mean_roc_auc_scores'][feature_idx], "\t",
    #     roc_auc_score_mx[feature_idx]
    # )
feature_ranks = sorted(
    zip(
        fs_data['feature_' + args.fs_rank_method],
        fs_data['features_uniq'],
        r_get_gene_symbols(eset_gex, robjects.StrVector(fs_data['features_uniq']))
    ),
    reverse=True
)
feature_ranks = feature_ranks[:args.top_fs]
feature_names_fs = [x for _, x, _ in feature_ranks]
print('Num Features:', args.top_fs, '/', len(fs_data['features_uniq']))
tr_data = {
    'feature_names': feature_names_fs,
    'y_scores_all': [],
    'y_tests_all': [],
    'split_data': [],
}
gscv_tr_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC()),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.gscv_splits, test_size=0.2),
    scoring='roc_auc', return_train_score=False, n_jobs=args.gscv_jobs,
    verbose=args.gscv_verbose
)
feature_names_fs = robjects.StrVector(feature_names_fs)
if args.eset_tr:
    base.load("data/" + args.eset_tr + ".Rda")
    eset_gex = robjects.globalenv[args.eset_tr]
eset_gex = r_filter_eset(eset_gex, feature_names_fs)
split_count = 0
print_header = True
while split_count < args.tr_splits:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_samples_tr = math.ceil(len(relapse_samples) * round((1 - args.tr_cv_size), 2))
    num_samples_ts = len(relapse_samples) - num_samples_tr
    samples_tr = relapse_samples[:num_samples_tr] + norelapse_samples[:num_samples_tr]
    eset_gex_tr = r_filter_eset(eset_gex, robjects.NULL, samples_tr)
    samples_ts = relapse_samples[num_samples_tr:(num_samples_tr + num_samples_ts)] + \
                 norelapse_samples[num_samples_tr:(num_samples_tr + num_samples_ts)]
    eset_gex_ts = r_filter_eset(eset_gex, robjects.NULL, samples_ts)
    X_train = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    if print_header:
        print(
            'TR:', num_samples_tr, '/', num_samples_tr,
            'CV:', num_samples_ts, '/', num_samples_ts,
        )
        print_header = False
    y_score = gscv_tr_clf.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_score)
    tr_data['y_scores_all'].extend(y_score.tolist())
    tr_data['y_tests_all'].extend(y_test.tolist())
    tr_data['split_data'].append({
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(svc.coef_[0]),
        'y_scores': y_score,
        'y_tests': y_test,
        'roc_auc_score': roc_auc,
    })
    split_count += 1
    print('TR splits:', '%3s' % split_count, 'ROC AUC:', '%.4f' % roc_auc)
# end while
for rank, feature, symbol in feature_ranks: print(feature, "\t", symbol, "\t", rank)
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
