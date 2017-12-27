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
r_filter_eset_relapse_labels = robjects.globalenv["filterEsetRelapseLabels"]
r_get_gene_symbols = robjects.globalenv["getGeneSymbols"]
r_get_dfx_features = robjects.globalenv["getDfxFeatures"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--fs-folds', type=int, default=100, help='num fs folds')
parser.add_argument('--cv-folds', type=int, default=100, help='num cv folds')
parser.add_argument('--cv-size', type=float, default=.33, help="cv size")
parser.add_argument('--relapse-fs-percent', type=float, default=.15, help='feature selection relapse percentage')
parser.add_argument('--min-dfx-fs', type=int, default=10, help='min num dfx features to select')
parser.add_argument('--top-fs', type=int, default=20, help='num top scoring features to select')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--fs-rank-method', type=str, default='mean_coefs', help="mean_coefs or mean_roc_auc_scores")
parser.add_argument('--gscv-folds', type=int, default=10, help='num gridsearchcv folds')
parser.add_argument('--gscv-jobs', type=int, default=-1, help="num gridsearchcv parallel jobs")
parser.add_argument('--gscv-verbose', type=int, default=1, help="gridsearchcv verbosity")
args = parser.parse_args()
base.load("data/eset_gex_nci_japan_luad.Rda")
eset_gex = robjects.globalenv["eset_gex_nci_japan_luad"]
fs_data = {
    'features_all': [],
    'fold_data': [],
}
fold_count = 0
low_fs_count = 0
while fold_count < args.fs_folds:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_relapse_samples_fs = math.ceil(len(relapse_samples) * args.relapse_fs_percent)
    num_norelapse_samples_fs = len(norelapse_samples) - len(relapse_samples) + num_relapse_samples_fs
    samples_fs = relapse_samples[:num_relapse_samples_fs] + \
                 norelapse_samples[:num_norelapse_samples_fs]
    features = r_get_dfx_features(r_filter_eset(eset_gex, robjects.NULL, samples_fs))
    if len(features) < args.min_dfx_fs:
        low_fs_count += 1
        continue
    num_samples_tr = len(relapse_samples) - (num_relapse_samples_fs * 2)
    samples_tr = relapse_samples[num_relapse_samples_fs:(num_relapse_samples_fs + num_samples_tr)] + \
                 norelapse_samples[num_norelapse_samples_fs:(num_norelapse_samples_fs + num_samples_tr)]
    eset_gex_tr = r_filter_eset(eset_gex, features, samples_tr)
    X_train = np.array(base.t(biobase.exprs(eset_gex_tr)))
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    samples_ts = relapse_samples[(num_relapse_samples_fs + num_samples_tr):] + \
                 norelapse_samples[(num_norelapse_samples_fs + num_samples_tr):]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    if args.svm_alg == 'liblinear':
        svc = LinearSVC()
    elif args.svm_alg == 'libsvm':
        svc = SVC(kernel='linear', cache_size=args.svm_cache_size)
    y_score = svc.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    fs_data['features_all'].extend(np.array(features).tolist())
    fs_data['fold_data'].append({
        'features': np.array(features).tolist(),
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(svc.coef_[0]),
        'y_scores': y_score,
        'y_tests': y_test,
        'roc_auc_score': roc_auc_score(y_test, y_score),
    })
    fold_count += 1
    print('FS Folds:', fold_count, 'Fails:', low_fs_count, end='\r', flush=True)
# end while
print('FS Folds:', fold_count, 'Fails:', low_fs_count)
# rank features
fs_data['features_uniq'] = natsorted(list(set(fs_data['features_all'])))
print(*fs_data['features_uniq'], sep="\n")
feature_mx_idx = {}
for idx, feature in enumerate(fs_data['features_uniq']): feature_mx_idx[feature] = idx
coef_mx = np.zeros((len(fs_data['features_uniq']), args.fs_folds), dtype="float64")
for fold_idx in range(len(fs_data['fold_data'])):
    fold_data = fs_data['fold_data'][fold_idx]
    for feature_idx in range(len(fold_data['features'])):
        coef_mx[feature_mx_idx[fold_data['features'][feature_idx]]][fold_idx] = \
            fold_data['coefs'][feature_idx]
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
roc_auc_score_mx = np.zeros((len(fs_data['features_uniq']), args.fs_folds), dtype="float64")
for fold_idx in range(len(fs_data['fold_data'])):
    fold_data = fs_data['fold_data'][fold_idx]
    for feature in fold_data['features']:
        roc_auc_score_mx[feature_mx_idx[feature]][fold_idx] = \
            fold_data['roc_auc_score']
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
features = [x for _, x, _ in feature_ranks]
cv_data = {
    'features': features,
    'y_scores_all': [],
    'y_tests_all': [],
    'fold_data': [],
}
features = robjects.StrVector(features)
gscv_cv_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('svc', LinearSVC()),
    ]),
    param_grid=[
        # { 'svc__kernel': ['linear'], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
        { 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] },
    ],
    cv=StratifiedShuffleSplit(n_splits=args.gscv_folds, test_size=0.2),
    scoring='roc_auc', return_train_score=False, n_jobs=args.gscv_jobs,
    verbose=args.gscv_verbose
)
fold_count = 0
while fold_count < args.cv_folds:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_samples_tr = math.ceil(len(relapse_samples) * round((1 - args.cv_size), 2))
    samples_tr = relapse_samples[:num_samples_tr] + norelapse_samples[:num_samples_tr]
    eset_gex_tr = r_filter_eset(eset_gex, features, samples_tr)
    X_train = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    samples_ts = relapse_samples[num_samples_tr:] + norelapse_samples[num_samples_tr:]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    y_score = gscv_cv_clf.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    cv_data['y_scores_all'].extend(y_score.tolist())
    cv_data['y_tests_all'].extend(y_test.tolist())
    cv_data['fold_data'].append({
        'fprs': fpr,
        'tprs': tpr,
        'thres': thres,
        'coefs': np.square(svc.coef_[0]),
        'y_scores': y_score,
        'y_tests': y_test,
        'roc_auc_score': roc_auc_score(y_test, y_score),
    })
    fold_count += 1
    print('CV Folds:', fold_count, end='\r', flush=True)
# end while
print('CV Folds:', fold_count)
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
plt.ylim([1.01,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
