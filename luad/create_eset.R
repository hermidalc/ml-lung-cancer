#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("impute"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))

cmd_args <- commandArgs(trailingOnly=TRUE)
for (file in cmd_args) {
    dir <- dirname(file)
    file_name_parts <- strsplit(basename(file), split="_", fixed=TRUE)[[1]]
    dataset_name <- file_name_parts[1]
    norm_type <- file_name_parts[2]
    if (length(file_name_parts) == 4) {
        eset_name <- paste0(c("eset", file_name_parts[1:2]), collapse="_")
        id_type <- "none"
    }
    else {
        eset_name <- paste0(c("eset", file_name_parts[1:3]), collapse="_")
        id_type <- file_name_parts[3]
    }
    print(paste("Creating:", eset_name), quote=FALSE, row.names=FALSE)
    eset <- ExpressionSet(
        assayData=as.matrix(column_to_rownames(as.data.frame(read_excel(
            file,
            sheet=3,
            col_names=TRUE,
            trim_ws=TRUE
        )), var="ID_REF")),
        phenoData=AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
            file,
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
    }
    else if (dataset_name %in% c("gse67639")) {
        annotation(eset) <- "hgu133plus2hsentrezg"
        geneSymbols <- getSYMBOL(probeset_ids, "hgu133plus2hsentrezg.db")
    }
    featureData(eset) <- AnnotatedDataFrame(data.frame(Symbol=geneSymbols))
    assign(eset_name, eset)
    save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
}
