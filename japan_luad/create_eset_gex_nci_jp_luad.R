#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
suppressPackageStartupMessages(library("genefilter"))
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/AffyU133Plus2array_NCC_226ADC_16Normal_MAS5normalized_reformatted.xlsx"
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    range = cell_cols("A:HS"),
    col_names = TRUE,
    trim_ws = TRUE
)), var="Probeset ID"))
pheno <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    range = cell_rows(1:227),
    col_names = TRUE,
    trim_ws = TRUE
)), var="Sample ID"))
# build eset
eset_gex_nci_japan_luad <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# annotate eset
probesetIds <- featureNames(eset_gex_nci_japan_luad)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_nci_japan_luad) <- data.frame(Symbol=geneSymbols)
# filter out control probesets
eset_gex_nci_japan_luad <- featureFilter(eset_gex_nci_japan_luad,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
save(eset_gex_nci_japan_luad, file="data/eset_gex_nci_japan_luad.Rda")
