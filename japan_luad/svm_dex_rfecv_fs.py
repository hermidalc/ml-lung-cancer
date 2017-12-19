#!/usr/bin/env python

import argparse
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import numpy2ri
# import pandas as pd
import numpy as np
from sklearn.model_selection import test_train_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
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
r_filter_eset = robjects.globalenv["filterEset"]
r_filter_eset_relapse_labels = robjects.globalenv["filterEsetRelapseLabels"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--num-folds', type=int, default=1000, help='num folds')
parser.add_argument('--num-rfecv-jobs', , type=int, default=1, help='num RFECV jobs'))
parser.add_argument('--relapse-fs-percent', type=float, default=.15, help='feature selection relapse percentage')
parser.add_argument('--min-num-features', type=int, default=10, help='feature selection minimum number of features')
args = parser.parse_args()

X = np.array(base.t(biobase.exprs(eset_gex)))
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
y = np.array(r_filter_eset_relapse_labels(eset_gex))
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, random_state=1301, stratify=y, test_size=0.33
)
svc = SVC(kernel='linear', cache_size=1000)
rfecv = RFECV(
    estimator=svc, step=10,
    cv=StratifiedKFold(y_train, shuffle=True, n_folds=3, random_state=1301),
    scoring='roc_auc', n_jobs=args.num_rfecv_jobs
)
rfecv.fit(X_train, y_train)
print('The optimal number of features is {}'.format(rfecv.n_features_))
features = [f for f,s in zip(X_train.columns, rfecv.support_) if s]
print('The selected features are:')
print ('{}'.format(features))
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (roc auc)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# y_score = rfecv.fit(X_scaled, y).decision_function(X_test_scaled)
# fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
# # plot ROC
# fpr, tpr, thres = roc_curve(y_test_all.flatten(), y_score_all.flatten(), pos_label=1)
# roc_auc_score = roc_auc_score(y_test_all.flatten(), y_score_all.flatten())
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.2f)' % roc_auc_score)
# plt.plot([0,1], [0,1], color='navy', lw=4, linestyle='--')
# plt.xlim([0.0,1.0])
# plt.ylim([0.0,1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()
