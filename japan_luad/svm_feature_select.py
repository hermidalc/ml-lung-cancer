#!/usr/bin/env python

import argparse
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import numpy2ri
# import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
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
args = parser.parse_args()
features_all = np.array([], dtype="str")
fpr_all = np.array([], dtype="float64")
tpr_all = np.array([], dtype="float64")
thres_all = np.array([], dtype="float64")
y_score_all = np.array([], dtype="float64")
y_test_all = np.array([], dtype="int")
roc_auc_score_all = np.array([], dtype="float64")
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
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    samples_ts = relapse_samples[(num_relapse_samples_fs + num_samples_tr):] + \
                 norelapse_samples[(num_norelapse_samples_fs + num_samples_tr):]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    svc = svm.SVC(kernel='linear', cache_size=1000)
    y_score = svc.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    features_all = np.concatenate((features_all, np.array(features)))
    fpr_all = np.concatenate((fpr_all, fpr))
    tpr_all = np.concatenate((tpr_all, tpr))
    thres_all = np.concatenate((thres_all, thres))
    y_score_all = np.concatenate((y_score_all, y_score))
    y_test_all = np.concatenate((y_test_all, y_test))
    roc_auc_score_all = np.append(roc_auc_score_all, roc_auc_score(y_test, y_score))
    fold_count += 1
    print('Folds:', fold_count, 'Fails:', low_fs_count, end='\r', flush=True)
# end while
print('Folds:', fold_count, 'Fails:', low_fs_count)
# save numpy arrays
np.save('data/features_all', features_all)
np.save('data/fpr_all', fpr_all)
np.save('data/tpr_all', tpr_all)
np.save('data/thres_all', thres_all)
np.save('data/y_score_all', y_score_all)
np.save('data/y_test_all', y_test_all)
np.save('data/roc_auc_score_all', roc_auc_score_all)
# plot ROC
fpr, tpr, thres = roc_curve(y_test_all.flatten(), y_score_all.flatten(), pos_label=1)
roc_auc_score = roc_auc_score(y_test_all.flatten(), y_score_all.flatten())
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
