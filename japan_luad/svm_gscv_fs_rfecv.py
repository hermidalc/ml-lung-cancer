#!/usr/bin/env python

import argparse, math, statistics, time
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
parser.add_argument('--gscv-folds', type=int, default=4, help='num gridsearchcv folds')
parser.add_argument('--gscv-jobs', type=int, default=1, help="num gridsearchcv parallel jobs")
parser.add_argument('--gscv-verbose', type=int, default=2, help="gridsearchcv verbosity")
parser.add_argument('--rfecv-folds', type=int, default=4, help='num rfecv folds')
parser.add_argument('--rfecv-jobs', type=int, default=-1, help="num rfecv parallel jobs")
parser.add_argument('--rfecv-step', type=float, default=0.01, help="rfecv step")
parser.add_argument('--rfecv-verbose', type=int, default=1, help="rfecv verbosity")
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--eset-src', type=str, default="eset_gex_gse31210", help="R eset for building svm")
parser.add_argument('--eset-cv', type=str, help="R eset for cross validation")
args = parser.parse_args()
grid_clf = GridSearchCV(
    Pipeline([
        ('slr', StandardScaler()),
        ('rfe',
            RFECV(
                LinearSVC(class_weight='balanced'), step=args.rfecv_step,
                cv=StratifiedShuffleSplit(n_splits=args.rfecv_folds, test_size=0.2),
                scoring='roc_auc', n_jobs=args.rfecv_jobs, verbose=args.rfecv_verbose
            )
        ),
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
base.load("data/" + args.eset_src + ".Rda")
eset_gex = robjects.globalenv[args.eset_src]
X = np.array(base.t(biobase.exprs(eset_gex)))
y = np.array(r_filter_eset_relapse_labels(eset_gex))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
start_time = time.time()
y_score = grid_clf.fit(X_train, y_train).decision_function(X_test)
print("Completed in %s minutes" % (math.ceil((time.time() - start_time) / 60)))
print(grid_clf.best_params_)
print(grid_clf.best_estimator_.named_steps['rfe'].n_features_)
feature_idxs = grid_clf.best_estimator_.named_steps['rfe'].get_support(indices=True)
feature_names = np.array(biobase.featureNames(eset_gex))
feature_names = feature_names[feature_idxs]
rankings = grid_clf.best_estimator_.named_steps['rfe'].ranking_[feature_idxs]
for feature, rank in zip(feature_names, rankings): print(feature, "\t", rank)
# save data
joblib.dump(grid_clf, 'data/svm_gscv_fs_rfecv.pkl')
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
roc_auc_score = roc_auc_score(y_test, y_score)
plt.rcParams['font.size'] = 24
plt.plot([0,1], [0,1], color='darkred', lw=2, linestyle='--', alpha=.8, label='Chance')
plt.plot(fpr, tpr, color='darkblue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_score)
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show(block=False)
