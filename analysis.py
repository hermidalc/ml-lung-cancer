#!/usr/bin/env python

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# import pandas as pd
# from rpy2.robjects import pandas2ri

base = importr('base')
utils = importr('utils')
# read Excel data
readxl = importr('readxl')
cellranger = importr('cellranger')
datafile = '/home/hermidalc/data/nci-lhc-nsclc/japan_luad/AffyU133+2array_NCC_226ADC_16Normal_MAS5normalized_test.xlsx'
data = readxl.read_excel(datafile, \
    sheet = 2, \
    col_names = True, \
    range = cellranger.cell_limits( \
        base.c(18, 2), \
        base.c(robjects.NA_Logical, robjects.NA_Logical) \
    )
)
print(utils.head(data))
# convert tibble to expressionset
biobase = importr('Biobase')
eset = biobase.ExpressionSet(assayData = base.as_matrix(data))
# limma analysis
limma = importr('limma')
print(utils.head(eset))
