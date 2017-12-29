#!/usr/bin/env R

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(library("annotate"))
suppressPackageStartupMessages(library("genefilter"))
# GSE31210
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
eset_gex_gse31210 <- eset_gex_gse31210[, eset_gex_gse31210$"Exclude Prognosis Analysis Incomplete Resection/Adjuvant Therapy" == 0]
# annotate eset
probesetIds <- featureNames(eset_gex_gse31210)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse31210) <- data.frame(Symbol=geneSymbols)
# filter eset
pData(eset_gex_gse31210) <- pData(eset_gex_gse31210)[,c("Relapse","Sex","Batch")]
# GSE30219
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/validation_datasets/GSE30219_series_matrix.xlsx"
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
eset_gex_gse30219 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse30219 <- eset_gex_gse30219[, eset_gex_gse30219$Sample_source_name_ch1 == "Lung Tumour"]
eset_gex_gse30219 <- eset_gex_gse30219[, eset_gex_gse30219$Histology == "ADC"]
eset_gex_gse30219 <- eset_gex_gse30219[, eset_gex_gse30219$"PT Stage" %in% c("T1","T2")]
pData(eset_gex_gse30219) <- pData(eset_gex_gse30219)[,c("Relapse","Sex","Batch")]
# annotate eset
probesetIds <- featureNames(eset_gex_gse30219)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse30219) <- data.frame(Symbol=geneSymbols)
# GSE37745
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/validation_datasets/GSE37745_series_matrix.xlsx"
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
eset_gex_gse37745 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Histology == "adeno"]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$"Tumor Stage" %in% c("1a","1b","2a","2b")]
eset_gex_gse37745 <- eset_gex_gse37745[, eset_gex_gse37745$Relapse %in% c(0,1)]
pData(eset_gex_gse37745) <- pData(eset_gex_gse37745)[c("Relapse","Sex","Batch")]
# annotate eset
probesetIds <- featureNames(eset_gex_gse37745)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse37745) <- data.frame(Symbol=geneSymbols)
# GSE50081
datafile <- "/home/hermidalc/data/nci-lhc-nsclc/japan_luad/validation_datasets/GSE50081_series_matrix.xlsx"
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
eset_gex_gse50081 <- ExpressionSet(
    assayData = exprs,
    phenoData = pheno,
    annotation="hgu133plus2"
)
# filter eset
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$Histology == "adenocarcinoma"]
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$"Tumor Stage" %in% c("1A","1B","2A","2B")]
eset_gex_gse50081 <- eset_gex_gse50081[, eset_gex_gse50081$Relapse %in% c(0,1)]
pData(eset_gex_gse50081) <- pData(eset_gex_gse50081)[c("Relapse","Sex","Batch")]
# annotate eset
probesetIds <- featureNames(eset_gex_gse50081)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset_gex_gse50081) <- data.frame(Symbol=geneSymbols)
# merge
eset_gex_merged <- combine(eset_gex_gse31210, eset_gex_gse30219)
eset_gex_merged <- combine(eset_gex_merged, eset_gex_gse37745)
eset_gex_merged <- combine(eset_gex_merged, eset_gex_gse50081)
# save
save(eset_gex_merged, file="data/eset_gex_merged.Rda")
