#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("impute"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))

cmd_args <- commandArgs(trailingOnly=TRUE)
for (dataset_name in cmd_args) {
    print(paste("Creating eset:", dataset_name))
    datafile <- paste0("data/", dataset_name, "_series_matrix.xlsx")
    eset <- ExpressionSet(
        assayData=as.matrix(column_to_rownames(as.data.frame(read_excel(
            datafile,
            sheet=3,
            col_names=TRUE,
            trim_ws=TRUE
        )), var="ID_REF")),
        phenoData=AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
            datafile,
            sheet=2,
            col_names=TRUE,
            trim_ws=TRUE
        )), var="Sample_geo_accession"))
    )
    # filter eset
    if (dataset_name == "gse31210") {
        eset <- eset[,eset$"Exclude Prognosis Analysis Incomplete Resection/Adjuvant Therapy" == 0]
    }
    else if (dataset_name == "gse8894") {
        eset <- eset[,eset$Histology == "ADC"]
    }
    else if (dataset_name == "gse30219") {
        eset <- eset[,eset$Histology == "ADC"]
        eset <- eset[,eset$Stage %in% c("1","1A","1B","2","2A","2B")]
    }
    else if (dataset_name == "gse37745") {
        eset <- eset[,eset$Histology == "ADC"]
        eset <- eset[,eset$Stage %in% c("1","1A","1B","2","2A","2B")]
        eset <- eset[,eset$Class %in% c(0,1)]
    }
    else if (dataset_name == "gse50081") {
        eset <- eset[,eset$Histology == "ADC"]
        eset <- eset[,eset$Stage %in% c("1","1A","1B","2","2A","2B")]
        eset <- eset[,eset$Class %in% c(0,1)]
    }
    else if (dataset_name == "gse67639") {
        eset <- eset[,eset$Histology == "ADC"]
        eset <- eset[,eset$Stage %in% c("1","1A","1B","2","2A","2B")]
        eset <- eset[,eset$"Exclude_incomplete_resection_adjuvant_therapy" == 0]
        impute_results <- impute.knn(exprs(eset), k=15)
        exprs(eset) <- impute_results$data
    }
    # annotate eset
    probeset_ids <- featureNames(eset)
    if (dataset_name %in% c("gse31210","gse8894","gse30219","gse37745","gse50081")) {
        annotation(eset) <- "hgu133plus2"
        geneSymbols <- getSYMBOL(probeset_ids, "hgu133plus2.db")
        eset_name <- paste0(c("eset", dataset_name), collapse="_")
    }
    else if (dataset_name %in% c("gse67639")) {
        annotation(eset) <- "hgu133plus2hsentrezg"
        geneSymbols <- getSYMBOL(probeset_ids, "hgu133plus2hsentrezg.db")
        eset_name <- paste0(c("eset", dataset_name, "gene"), collapse="_")
    }
    featureData(eset) <- AnnotatedDataFrame(data.frame(Symbol=geneSymbols))
    assign(eset_name, eset)
    save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
}
