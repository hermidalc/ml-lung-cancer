#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
suppressPackageStartupMessages(library("genefilter"))
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/validation_datasets/GSE37745_series_matrix.xlsx"
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    col_names = TRUE,
    trim_ws = TRUE
)), var="ID_REF"))
pData <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Sample_geo_accession"))
# build eset
eset_gex_gse37745 <- ExpressionSet(
    assayData = exprs,
    phenoData = pData,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Histology == "adeno"]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$"Tumor Stage" %in% c("1a","1b","2a","2b")]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Relapse %in% c(0,1)]
# annotate eset
probesetIds <- featureNames(eset_gex_gse37745)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse37745) <- data.frame(Symbol=geneSymbols)
# filter out control probesets
eset_gex_gse37745 <- featureFilter(eset_gex_gse37745,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
save(eset_gex_gse37745, file="data/eset_gex_gse37745.Rda")
