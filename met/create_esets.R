#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
source("config.R")

for (dataset_name in dataset_names) {
    exprs_file_basename <- paste0(c(dataset_name, "data"), collapse="_")
    pdata_file_basename <- paste0(c(dataset_name, "meta"), collapse="_")
    eset_name <- paste0(c("eset", dataset_name), collapse="_")
    cat("Creating:", eset_name, "\n")
    pdata <- read.delim(paste0("data/", pdata_file_basename, ".txt"), row.names=1)
    pdata <- pdata[!is.na(pdata$Class),]
    eset <- ExpressionSet(
        assayData=t(read.delim(paste0("data/", exprs_file_basename, ".txt"), row.names=1)),
        phenoData=AnnotatedDataFrame(pdata)
    )
    assign(eset_name, eset)
    save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
}
