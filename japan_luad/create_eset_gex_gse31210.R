#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
suppressPackageStartupMessages(library("genefilter"))
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/GSE31210_series_matrix_gcrma.xlsx"
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    col_names = TRUE,
    trim_ws = TRUE
)), var="ID_REF"))
pheno <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Sample ID"))
# build eset
eset_gex_gse31210 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse31210 <- eset_gex_gse31210[, eset_gex_gse31210$"Exclude Prognosis Analysis Incomplete Resection/Adjuvant Therapy" == 1]
# annotate eset
probesetIds <- featureNames(eset_gex_gse31210)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse31210) <- data.frame(Symbol=geneSymbols)
# filter out control probesets
eset_gex_gse31210 <- featureFilter(eset_gex_gse31210,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
save(eset_gex_gse31210, file="data/eset_gex_gse31210.Rda")
