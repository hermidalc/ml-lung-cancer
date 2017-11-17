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
base.load("eset_gex.Rda")
eset = robjects.globalenv["eset.gex"]
base.source("functions.R")
r_rand_perm_sample_nums = robjects.globalenv["randPermSampleNums"]
r_filter_eset_s = robjects.globalenv["filterEsetSamples"]
r_filter_eset_f = robjects.globalenv["filterEsetFeatures"]
r_select_exp_features = robjects.globalenv["selectExpFeatures"]
relapse_fs_percent = .15
for i in range(0, 1):
    relapse_sample_nums = r_rand_perm_sample_nums(eset, 1)
    norelapse_sample_nums = r_rand_perm_sample_nums(eset, 0)
    num_fs_relapse_samples = math.ceil(len(relapse_sample_nums) * relapse_fs_percent)
    num_fs_norelapse_samples = len(norelapse_sample_nums) - len(relapse_sample_nums) + num_fs_relapse_samples
    eset_fs = r_filter_eset_s(eset, relapse_sample_nums[:num_fs_relapse_samples], norelapse_sample_nums[:num_fs_norelapse_samples])
    features = r_select_exp_features(eset_fs)
