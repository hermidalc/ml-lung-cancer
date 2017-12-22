#!/usr/bin/env python

import argparse, math, statistics
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
r_get_diff_exp_features = robjects.globalenv["getDiffExpFeatures"]
# config
parser = argparse.ArgumentParser()
parser.add_argument('--num-folds', type=int, default=10, help='num folds')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='libsvm', help="svm algorithm (libsvm or liblinear)")
args = parser.parse_args()
base.load("data/eset_gex_nci_japan_luad.Rda")
eset_gex = robjects.globalenv["eset_gex_nci_japan_luad"]
gs_param_grid = [
    # { 'svc__kernel': ['linear'], 'svc__C': [0.01, 0.1, 1, 10, 100, 1000] },
    { 'svc__C': [0.01, 0.1, 1, 10, 100, 1000] },
]
pipe = Pipeline([
    ('slr', StandardScaler()),
    ('rfe',
        RFECV(
            LinearSVC(class_weight='balanced'), step=0.05,
            cv=StratifiedShuffleSplit(n_splits=args.num_folds, test_size=0.2),
            scoring='roc_auc', verbose=1
        )
    ),
    ('svc', LinearSVC(class_weight='balanced')),
])
clf = GridSearchCV(
    pipe, param_grid=gs_param_grid,
    cv=StratifiedShuffleSplit(n_splits=args.num_folds, test_size=0.2),
    scoring='roc_auc', n_jobs=-1, verbose=2
)
X = np.array(base.t(biobase.exprs(eset_gex)))
y = np.array(r_filter_eset_relapse_labels(eset_gex))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
print(clf.best_params_)
print(clf.ranking_)
# save data
joblib.dump(clf, 'data/svm_gscv_fs_rfecv.pkl')
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
roc_auc_score = roc_auc_score(y_test, y_score)
plt.rcParams['font.size'] = 24
plt.plot([0,1], [0,1], color='darkred', lw=4, linestyle='--', alpha=.8, label='Chance')
plt.plot(fpr, tpr, color='darkblue', lw=4, label='ROC curve (area = %0.4f)' % roc_auc_score)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
