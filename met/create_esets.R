#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--datasets", type="character", nargs="+", help="datasets")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
args <- parser$parse_args()
if (!is.null(args$datasets)) {
    dataset_names <- intersect(dataset_names, args$datasets)
}
if (!is.null(args$data_type)) {
    data_types <- intersect(data_types, args$data_type)
}
if (!is.null(args$norm_meth)) {
    norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
}
if (!is.null(args$feat_type)) {
    feat_types <- feat_types[feat_types %in% args$feat_type]
}
for (dataset_name in dataset_names) {
    for (data_type in data_types) {
        suffixes <- c(data_type)
        pdata_file_basename <- paste0(c(dataset_name, suffixes, "meta"), collapse="_")
        pdata_file <- paste0("data/", pdata_file_basename, ".txt")
        for (norm_meth in norm_methods) {
            for (feat_type in feat_types) {
                suffixes <- c(data_type)
                for (suffix in c(norm_meth, feat_type)) {
                    if (suffix != "none") suffixes <- c(suffixes, suffix)
                }
                exprs_file_basename <- paste0(c(dataset_name, suffixes, "data"), collapse="_")
                exprs_file <- paste0("data/", exprs_file_basename, ".txt")
                if (file.exists(pdata_file) && file.exists(exprs_file)) {
                    if (!exists("pdata")) {
                        cat("Loading:", pdata_file_basename, "\n")
                        pdata <- read.delim(pdata_file, row.names=1)
                        pdata <- pdata[!is.na(pdata$Class),]
                    }
                    eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
                    cat("Creating:", eset_name, "\n")
                    eset <- ExpressionSet(
                        assayData=t(read.delim(exprs_file, row.names=1)),
                        phenoData=AnnotatedDataFrame(pdata)
                    )
                    assign(eset_name, eset)
                    save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
                }
            }
        }
        if (exists("pdata")) remove(pdata)
    }
}
