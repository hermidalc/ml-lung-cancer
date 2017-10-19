#!/usr/bin/env python

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
base.load("eset.Rda")
eset = robjects.globalenv['eset']
base.source("feature_selection.R")
r_select_gex_features = robjects.globalenv['selectGeneExpressionFeatures']
for i in range(0, 0):
    table = r_select_gex_features(eset)
    print(base.summary(table))
    print(i)

svc = svm.SVC(kernel='linear')
