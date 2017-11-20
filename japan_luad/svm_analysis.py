#!/usr/bin/env python

import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm

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
num_perms = 50
relapse_fs_percent = .15
for i in range(1, num_perms):
    relapse_samples = r_rand_perm_sample_nums(eset_gex, True)
    norelapse_samples = r_rand_perm_sample_nums(eset_gex, False)
    num_relapse_samples_fs = math.ceil(len(relapse_samples) * relapse_fs_percent)
    num_norelapse_samples_fs = len(norelapse_samples) - len(relapse_samples) + num_relapse_samples_fs
    samples_fs = relapse_samples[:num_relapse_samples_fs] + \
                 norelapse_samples[:num_norelapse_samples_fs]
    eset_gex_fs = r_filter_eset(eset_gex, robjects.NULL, samples_fs)
    features_df = r_select_exp_features(eset_gex_fs)
    features = base.rownames(features_df)
    num_samples_tr = len(relapse_samples) - (num_relapse_samples_fs * 2)
    samples_tr = relapse_samples[num_relapse_samples_fs:(num_relapse_samples_fs + num_samples_tr)] + \
                 norelapse_samples[num_norelapse_samples_fs:(num_norelapse_samples_fs + num_samples_tr)]
    eset_gex_tr = r_filter_eset(eset_gex, features, samples_tr)
    X_tr = np.array(base.t(biobase.exprs(eset_gex_tr)))
    y_tr = np.array(r_filter_eset_relapse_labels(eset_gex_tr))
    clf = svm.SVC(kernel='linear')
    clf.fit(X_tr, y_tr)
    samples_ts = relapse_samples[(num_relapse_samples_fs + num_samples_tr):] + \
                 norelapse_samples[(num_norelapse_samples_fs + num_samples_tr):]
    eset_gex_ts = r_filter_eset(eset_gex, features, samples_ts)
    X_ts = np.array(base.t(biobase.exprs(eset_gex_ts)))
    y_ts = np.array(r_filter_eset_relapse_labels(eset_gex_ts))
    print(clf.predict(X_ts))
