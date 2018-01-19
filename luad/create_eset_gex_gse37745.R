#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
datafile <- "data/GSE37745_series_matrix.xlsx"
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
eset_gex_gse37745 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Histology == "adeno"]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Stage %in% c("1a","1b","2a","2b")]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Relapse %in% c(0,1)]
# annotate eset
probesetIds <- featureNames(eset_gex_gse37745)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse37745) <- data.frame(Symbol=geneSymbols)
save(eset_gex_gse37745, file="data/eset_gex_gse37745.Rda")
