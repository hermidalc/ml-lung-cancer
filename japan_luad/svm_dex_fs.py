#!/usr/bin/env python

import argparse, math, statistics, pickle
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import numpy2ri
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
plt.rcParams['font.size'] = 24
base = importr("base")
biobase = importr("Biobase")
base.load("data/eset_gex.Rda")
eset_gex = robjects.globalenv["eset.gex"]
base.source("functions.R")
r_rand_perm_sample_nums = robjects.globalenv["randPermSampleNums"]
r_filter_eset = robjects.globalenv["filterEset"]
r_filter_eset_relapse_labels = robjects.globalenv["filterEsetRelapseLabels"]
r_get_diff_exp_features = robjects.globalenv["getDiffExpFeatures"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--num-folds', type=int, default=1000, help='num folds')
parser.add_argument('--relapse-fs-percent', type=float, default=.15, help='feature selection relapse percentage')
parser.add_argument('--min-num-features', type=int, default=10, help='feature selection minimum number of features')
parser.add_argument('--num-top-features', type=int, default=10, help='feature selection number top scoring features')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='libsvm', help="svm algorithm (libsvm or liblinear)")
args = parser.parse_args()
# fs_features = np.array([], dtype="str")
# fs_fprs = np.array([], dtype="float64")
# fs_tprs = np.array([], dtype="float64")
# fs_thres = np.array([], dtype="float64")
# fs_y_scores = np.array([], dtype="float64")
# fs_y_tests = np.array([], dtype="int")
# fs_roc_auc_scores = np.array([], dtype="float64")
fs_data = {
    'features_flat': [],
    'fold_data': [],
}
fold_count = 0
low_fs_count = 0
while fold_count < args.num_folds:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_relapse_samples_fs = math.ceil(len(relapse_samples) * args.relapse_fs_percent)
    num_norelapse_samples_fs = len(norelapse_samples) - len(relapse_samples) + num_relapse_samples_fs
    samples_fs = relapse_samples[:num_relapse_samples_fs] + \
                 norelapse_samples[:num_norelapse_samples_fs]
    eset_gex_fs = r_filter_eset(eset_gex, robjects.NULL, samples_fs)
    features_df = r_get_diff_exp_features(eset_gex_fs)
    features = base.rownames(features_df)
    if len(features) < args.min_num_features:
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
    if args.svm_alg == 'libsvm':
        svc = SVC(kernel='linear', cache_size=args.svm_cache_size)
    elif args.svm_alg == 'liblinear':
        svc = LinearSVC()
    y_score = svc.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    fs_data['features_flat'].extend(np.array(features).tolist())
    fs_data['fold_data'].append({
        'features': np.array(features).tolist(),
        'fprs': fpr.tolist(),
        'tprs': tpr.tolist(),
        'thres': thres.tolist(),
        'coefs': svc.coef_[0].tolist(),
        'y_scores': y_score.tolist(),
        'y_tests': y_test.tolist(),
        'roc_auc_score': roc_auc_score(y_test, y_score),
    })
    fold_count += 1
    print('Folds:', fold_count, 'Fails:', low_fs_count, end='\r', flush=True)
# end while
print('Folds:', fold_count, 'Fails:', low_fs_count)
# rank features
fs_data['features_uniq'] = list(set(fs_data['features_flat']))
feature_mx_idx = {}
for idx, feature in enumerate(fs_data['features_uniq']): feature_mx_idx[feature] = idx
coef_mx = np.zeros((len(fs_data['features_uniq']), args.num_folds), dtype="float64")
for fold_idx in range(len(fs_data['fold_data'])):
    fold_data = fs_data['fold_data'][fold_idx]
    for feature_idx in range(len(fold_data['features'])):
        coef_mx[feature_mx_idx[fold_data['features'][feature_idx]]][fold_idx] = \
            abs(fold_data['coefs'][feature_idx])
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
roc_auc_score_mx = np.zeros((len(fs_data['features_uniq']), args.num_folds), dtype="float64")
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
# for y, x in sorted(zip(fs_data['feature_mean_roc_auc_scores'], fs_data['features_uniq']), reverse=True):
#     print(x, "\t", y)
features = [x for _, x in sorted(zip(fs_data['feature_mean_coefs'], fs_data['features_uniq']), reverse=True)]
features = robjects.StrVector(features[:10])
fl_data = {
    'y_scores_flat': [],
    'y_tests_flat': [],
    'fold_data': [],
}
fold_count = 0
while fold_count < args.num_folds:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_samples_tr = math.ceil(len(relapse_samples) * .80)
    samples_tr = relapse_samples[:num_samples_tr] + norelapse_samples[:num_samples_tr]
    eset_gex_tr = r_filter_eset(eset_gex, features, samples_tr)
    X_train = np.array(base.t(biobase.exprs(eset_gex_tr)))
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    samples_ts = relapse_samples[num_samples_tr:] + norelapse_samples[num_samples_tr:]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    if args.svm_alg == 'libsvm':
        svc = SVC(kernel='linear', cache_size=args.svm_cache_size)
    elif args.svm_alg == 'liblinear':
        svc = LinearSVC()
    y_score = svc.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    fl_data['y_scores_flat'].extend(y_score.tolist())
    fl_data['y_tests_flat'].extend(y_test.tolist())
    fl_data['fold_data'].append({
        'fprs': fpr.tolist(),
        'tprs': tpr.tolist(),
        'thres': thres.tolist(),
        'coefs': svc.coef_[0].tolist(),
        'y_scores': y_score.tolist(),
        'y_tests': y_test.tolist(),
        'roc_auc_score': roc_auc_score(y_test, y_score),
    })
    fold_count += 1
    print('Folds:', fold_count, 'Fails:', low_fs_count, end='\r', flush=True)
# end while
print('Folds:', fold_count, 'Fails:', low_fs_count)
# save data
# np.save('data/fs_features.npy', fs_features)
# np.save('data/fs_fprs', fs_fprs)
# np.save('data/fs_tprs', fs_tprs)
# np.save('data/fs_thres', fs_thres)
# np.save('data/fs_y_scores', fs_y_scores)
# np.save('data/fs_y_tests', fs_y_tests)
# np.save('data/fs_roc_auc_scores', fs_roc_auc_scores)
fs_data_fh = open('data/fs_data.pkl', 'wb')
fl_data_fh = open('data/fl_data.pkl', 'wb')
pickle.dump(fs_data, fs_data_fh, pickle.HIGHEST_PROTOCOL)
pickle.dump(fl_data, fl_data_fh, pickle.HIGHEST_PROTOCOL)
fs_data_fh.close()
fl_data_fh.close()
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(fl_data['y_tests_flat'], fl_data['y_scores_flat'], pos_label=1)
roc_auc_score = roc_auc_score(fl_data['y_tests_flat'], fl_data['y_scores_flat'])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.2f)' % roc_auc_score)
plt.plot([0,1], [0,1], color='navy', lw=4, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
