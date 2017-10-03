#!/usr/bin/env R

library('tibble')
library('readxl')
library('Biobase')
library('genefilter')
library('limma')
# read Excel data
datafile <- '/home/hermidalc/data/nci-lhc-nsclc/japan_luad/AffyU133+2array_NCC_226ADC_16Normal_MAS5normalized_test.xlsx'
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Probeset ID"))
pData <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Biology ID"))
# build ExpressionSet
eset <- ExpressionSet(
    assayData = exprs,
    phenoData = pData,
    annotation="hgu133plus2"
)
# filter control probesets

# limma analysis
design <- model.matrix(~0+factor(pData(eset)$'T/N'))
colnames(design) <- c("Normal", "Tumor")
fit <- lmFit(eset, design)
cont.matrix <- makeContrasts(TumorvsNormal=Tumor-Normal, levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)
topTable(fit2, adjust="BH")
