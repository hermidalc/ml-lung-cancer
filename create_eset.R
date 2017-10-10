#!/usr/bin/env R

library('tibble')
library('readxl')
library('Biobase')
library('hgu133plus2.db')
library('annotate')
library('genefilter')
datafile <- '/home/hermidalc/data/nci-lhc-nsclc/japan_luad/AffyU133Plus2array_NCC_226ADC_16Normal_MAS5normalized_reformatted.xlsx'
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    range = cell_cols("A:HS"),
    col_names = TRUE,
    trim_ws = TRUE
)), var="Probeset ID"))
pData <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    range = cell_rows(1:227),
    col_names = TRUE,
    trim_ws = TRUE
)), var="Biology ID"))
# build eset
eset <- ExpressionSet(
    assayData = exprs,
    phenoData = pData,
    annotation="hgu133plus2"
)
# annotate eset
probesetIds <- featureNames(eset)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset) <- data.frame(Symbol=geneSymbols)
# filter out control probesets
eset.filtered <- featureFilter(eset,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
save(eset.filtered, file="eset.Rda")
