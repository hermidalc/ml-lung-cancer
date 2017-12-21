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
from sklearn.metrics import auc, roc_curve, roc_auc_score
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
parser.add_argument('--data', type=str, help='input data')
parser.add_argument('--num-folds', type=int, default=1000, help='num folds')
parser.add_argument('--svm-cache-size', type=int, default=2000, help='svm cache size')
parser.add_argument('--svm-alg', type=str, default='liblinear', help="svm algorithm (liblinear or libsvm)")
parser.add_argument('--cv-test-size', type=float, default=.33, help="cv test size")
args = parser.parse_args()
base.load("data/eset_gex_gse50081.Rda")
eset_gex_cv = robjects.globalenv["eset_gex_gse50081"]
features = [
    '209251_x_at',
    '200092_s_at',
    '209365_s_at',
    '213646_x_at',
    '211296_x_at',
    '213084_x_at',
    '210427_x_at',
    '201049_s_at',
    '211972_x_at',
    '212363_x_at',
    '212661_x_at',
    '213347_x_at',
    '226675_s_at',
    '201090_x_at',
]
cv_data = {
    'features': features,
    'y_scores_all': [],
    'y_tests_all': [],
    'fold_data': [],
}
features = robjects.StrVector(features)
fold_count = 0
while fold_count < args.num_folds:
    relapse_samples = r_rand_perm_sample_nums(eset_gex_cv, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex_cv, False)
    num_samples_tr = math.ceil(len(relapse_samples) * round((1 - args.cv_test_size),2))
    samples_tr = relapse_samples[:num_samples_tr] + norelapse_samples[:num_samples_tr]
    eset_gex_cv_tr = r_filter_eset(eset_gex_cv, features, samples_tr)
    X_train = np.array(base.t(biobase.exprs(eset_gex_cv_tr)))
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = np.array(r_filter_eset_relapse_labels(eset_gex_cv_tr))
    samples_ts = relapse_samples[num_samples_tr:] + norelapse_samples[num_samples_tr:]
    eset_gex_cv_ts = r_filter_eset(eset_gex_cv, features, samples_ts)
    X_test = np.array(base.t(biobase.exprs(eset_gex_cv_ts)))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.array(r_filter_eset_relapse_labels(eset_gex_cv_ts))
    if args.svm_alg == 'libsvm':
        svc = SVC(kernel='linear', cache_size=args.svm_cache_size)
    elif args.svm_alg == 'liblinear':
        svc = LinearSVC()
    y_score = svc.fit(X_train_scaled, y_train).decision_function(X_test_scaled)
    fpr, tpr, thres = roc_curve(y_test, y_score, pos_label=1)
    cv_data['y_scores_all'].extend(y_score.tolist())
    cv_data['y_tests_all'].extend(y_test.tolist())
    cv_data['fold_data'].append({
        'fprs': fpr.tolist(),
        'tprs': tpr.tolist(),
        'thres': thres.tolist(),
        'coefs': svc.coef_[0].tolist(),
        'y_scores': y_score.tolist(),
        'y_tests': y_test.tolist(),
        'roc_auc_score': roc_auc_score(y_test, y_score),
    })
    fold_count += 1
    print('CV Folds:', fold_count, end='\r', flush=True)
# end while
print('CV Folds:', fold_count)
# save data
# cv_data_fh = open('data/cv_data.pkl', 'wb')
# pickle.dump(cv_data, cv_data_fh, pickle.HIGHEST_PROTOCOL)
# cv_data_fh.close()
# plot ROC AUC Curve
fpr, tpr, thres = roc_curve(cv_data['y_tests_all'], cv_data['y_scores_all'], pos_label=1)
roc_auc_score = roc_auc_score(cv_data['y_tests_all'], cv_data['y_scores_all'])
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
