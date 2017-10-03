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
meta_tbl = readxl.read_excel(
    datafile,
    sheet = 2,
    col_names = True,
    trim_ws = True
)
biobase = importr('Biobase')
# ph_adf = biobase.AnnotatedDataFrame(data = meta_tbl)
# print(biobase.pData(ph_adf))
data_tbl = readxl.read_excel(
    datafile,
    sheet = 3,
    col_names = True,
    trim_ws = True,
)
# convert tibbles to eset
eset = biobase.ExpressionSet(
    assayData = base.as_matrix(data_tbl),
    phenoData = biobase.AnnotatedDataFrame(meta_tbl)
)
# limma analysis
limma = importr('limma')
print(utils.head(eset))
