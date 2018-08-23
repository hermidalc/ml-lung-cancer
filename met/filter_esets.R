#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--filter", type="character", nargs="+", help="filter function")
parser$add_argument("--datasets", type="character", nargs="+", help="datasets")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
args <- parser$parse_args()
if (is.null(args$filter) || !(args$filter %in% c("cff"))) {
    stop("--filter option required")
}
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
if (args$filter == "cff") {
    for (dataset_name in dataset_names) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
                    eset_file <- paste0("data/", eset_name, ".Rda")
                    if (file.exists(eset_file)) {
                        cat("Loading: ", eset_name, "\n")
                        load(eset_file)
                        if (exists("common_feature_names")) {
                            common_feature_names <- intersect(common_feature_names, featureNames(get(eset_name)))
                        }
                        else {
                            common_feature_names <- featureNames(get(eset_name))
                        }
                    }
                }
            }
        }
    }
    if (args$load_only) quit()
    for (dataset_name in dataset_names) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
                    if (exists(eset_name)) {
                        eset_filtered_name <- paste0(c(eset_name, args$filter), collapse="_")
                        cat("Creating:", eset_filtered_name, "\n")
                        eset_filtered <- get(eset_name)
                        eset_filtered <- eset_filtered[common_feature_names,]
                        assign(eset_filtered_name, eset_filtered)
                        save(list=eset_filtered_name, file=paste0("data/", eset_filtered_name, ".Rda"))
                    }
                }
            }
        }
    }
}
