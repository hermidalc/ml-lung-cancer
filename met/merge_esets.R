#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--dataset-tr", type="character", nargs="+", help="dataset tr")
parser$add_argument("--num-combo-tr", type="integer", help="num tr datasets to combine")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
args <- parser$parse_args()
if (!is.null(args$dataset_tr)) {
    dataset_names <- intersect(dataset_names, args$dataset_tr)
}
if (!is.null(args$dataset_tr) && !is.null(args$num_combo)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$dataset_tr), as.integer(args$num_combo))
} else if (!is.null(args$dataset_tr)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$dataset_tr), length(args$dataset_tr))
} else {
    if (is.null(args$num_combo)) stop("--num-combo-tr option required")
    dataset_tr_name_combos <- combn(dataset_names, as.integer(args$num_combo))
}
if (!is.null(args$data_type)) {
    data_types <- intersect(data_types, args$data_type)
}
if (!is.null(args$norm_meth)) {
    norm_methods <- intersect(norm_methods, args$norm_meth)
}
if (!is.null(args$feat_type)) {
    feat_types <- intersect(feat_types, args$feat_type)
}
for (dataset_name in dataset_names) {
    for (data_type in data_types) {
        for (norm_meth in norm_methods) {
            for (feat_type in feat_types) {
                suffixes <- c(data_type)
                for (suffix in c(norm_meth, feat_type)) {
                    if (!(suffix %in% c("none", "None"))) suffixes <- c(suffixes, suffix)
                }
                eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
                eset_file <- paste0("data/", eset_name, ".Rda")
                if (file.exists(eset_file)) {
                    cat(" Loading:", eset_name, "\n")
                    load(eset_file)
                    # subset common pheno data
                    eset <- get(eset_name)
                    pData(eset) <- pData(eset)[common_pheno_names]
                    assign(eset_name, eset)
                }
            }
        }
    }
}
if (args$load_only) quit()
for (col in 1:ncol(dataset_tr_name_combos)) {
    for (data_type in data_types) {
        for (norm_meth in norm_methods) {
            for (feat_type in feat_types) {
                suffixes <- c(data_type)
                for (suffix in c(norm_meth, feat_type)) {
                    if (!(suffix %in% c("none", "None"))) suffixes <- c(suffixes, suffix)
                }
                for (dataset_name in dataset_tr_name_combos[,col]) {
                    eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
                    if (exists(eset_name)) {
                        if (exists("common_feature_names")) {
                            common_feature_names <- intersect(common_feature_names, featureNames(get(eset_name)))
                        }
                        else {
                            common_feature_names <- featureNames(get(eset_name))
                        }
                    }
                }
                eset_1_name <- paste0(c("eset", dataset_tr_name_combos[1,col], suffixes), collapse="_")
                eset_2_name <- paste0(c("eset", dataset_tr_name_combos[2,col], suffixes), collapse="_")
                if (exists(eset_1_name) && exists(eset_2_name)) {
                    eset_merged_name <- paste0(
                        c("eset", dataset_tr_name_combos[,col], suffixes, "mrg", "tr"), collapse="_"
                    )
                    cat("Creating:", eset_merged_name, "\n")
                    eset_1 <- get(eset_1_name)
                    eset_1 <- eset_1[common_feature_names,]
                    eset_2 <- get(eset_2_name)
                    eset_2 <- eset_2[common_feature_names,]
                    eset_merged <- combine(eset_1, eset_2)
                    if (nrow(dataset_tr_name_combos) > 2) {
                        for (row in 3:nrow(dataset_tr_name_combos)) {
                            eset_n_name <- paste0(
                                c("eset", dataset_tr_name_combos[row,col], suffixes), collapse="_"
                            )
                            eset_n <- get(eset_n_name)
                            eset_n <- eset_n[common_feature_names,]
                            eset_merged <- combine(eset_merged, eset_n)
                        }
                    }
                    assign(eset_merged_name, eset_merged)
                    save(list=eset_merged_name, file=paste0("data/", eset_merged_name, ".Rda"))
                    remove(list=c(eset_merged_name))
                }
                if (exists("common_feature_names")) remove(common_feature_names)
            }
        }
    }
}
