#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm

style.use("ggplot")

svc = svm.SVC(kernel='linear')
