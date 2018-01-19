#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
datafile <- "data/GSE50081_series_matrix.xlsx"
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
)), var="Sample_geo_accession"))
# build eset
eset_gex_gse50081 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$Histology == "adenocarcinoma"]
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$Stage %in% c("1A","1B","2A","2B")]
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$Relapse %in% c(0,1)]
# annotate eset
probesetIds <- featureNames(eset_gex_gse50081)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse50081) <- data.frame(Symbol=geneSymbols)
save(eset_gex_gse50081, file="data/eset_gex_gse50081.Rda")
