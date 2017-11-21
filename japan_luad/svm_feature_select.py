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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
base = importr("base")
biobase = importr("Biobase")
base.load("eset_gex.Rda")
eset_gex = robjects.globalenv["eset.gex"]
base.source("functions.R")
r_rand_perm_sample_nums = robjects.globalenv["randPermSampleNums"]
r_filter_eset = robjects.globalenv["filterEset"]
r_filter_eset_relapse_labels = robjects.globalenv["filterEsetRelapseLabels"]
r_select_exp_features = robjects.globalenv["selectExpFeatures"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--num-perms', type=int, default=1000, help='num permutations')
parser.add_argument('--relapse-fs-percent', type=float, default=.15, help='feature selection using percentage')
parser.add_argument('--min-num-features', type=int, default=10, help='minimum number features')
args = parser.parse_args()
all_y_scores = np.array([])
all_y_test = np.array([], dtype="int")
perm_count = 1
low_fs_count = 0
while perm_count < args.num_perms:
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_relapse_samples_fs = math.ceil(len(relapse_samples) * args.relapse_fs_percent)
    num_norelapse_samples_fs = len(norelapse_samples) - len(relapse_samples) + num_relapse_samples_fs
    samples_fs = relapse_samples[:num_relapse_samples_fs] + \
                 norelapse_samples[:num_norelapse_samples_fs]
    eset_gex_fs = r_filter_eset(eset_gex, robjects.NULL, samples_fs)
    features_df = r_select_exp_features(eset_gex_fs)
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
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train_scaled, y_train)
    samples_ts = relapse_samples[(num_relapse_samples_fs + num_samples_tr):] + \
                 norelapse_samples[(num_norelapse_samples_fs + num_samples_tr):]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_ts)))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    clf.predict(X_test_scaled)
    y_scores = clf.decision_function(X_test_scaled)
    all_y_scores = np.append(all_y_scores, y_scores)
    all_y_test = np.append(all_y_test, y_test)
    print('Permutations:', perm_count, 'FS fails:', low_fs_count, end='\r', flush=True)
    perm_count += 1

print("\n", roc_auc_score(all_y_test, all_y_scores))

    # fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=2)
    # metrics.auc(fpr, tpr)
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # n_classes = 0
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # plt.figure()
    # lw = 2
    # plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    # plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0,1.0])
    # plt.ylim([0.0,1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    # plt.legend(loc="lower right")
    # plt.show()
