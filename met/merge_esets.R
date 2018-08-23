#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--datasets", type="character", nargs="+", help="datasets")
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
args <- parser$parse_args()
if (!is.null(args$datasets) && !is.null(args$num_tr_combo)) {
    dataset_name_combos <- combn(intersect(dataset_names, args$datasets), as.integer(args$num_tr_combo))
} else if (!is.null(args$datasets)) {
    dataset_name_combos <- combn(intersect(dataset_names, args$datasets), length(args$datasets))
} else {
    if (is.null(args$num_tr_combo)) stop("--num-tr-combo option required")
    dataset_name_combos <- combn(dataset_names, as.integer(args$num_tr_combo))
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
for (col in 1:ncol(dataset_name_combos)) {
    for (data_type in data_types) {
        for (norm_meth in norm_methods) {
            for (feat_type in feat_types) {
                suffixes <- c(data_type)
                for (suffix in c(norm_meth, feat_type)) {
                    if (suffix != "none") suffixes <- c(suffixes, suffix)
                }
                for (dataset_name in dataset_name_combos[,col]) {
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
                eset_1_name <- paste0(c("eset", dataset_name_combos[1,col], suffixes), collapse="_")
                eset_2_name <- paste0(c("eset", dataset_name_combos[2,col], suffixes), collapse="_")
                if (exists(eset_1_name) && exists(eset_2_name)) {
                    eset_merged_name <- paste0(
                        c("eset", dataset_name_combos[,col], suffixes, "mrg", "tr"), collapse="_"
                    )
                    cat("Creating:", eset_merged_name, "\n")
                    eset_1 <- get(eset_1_name)
                    eset_1 <- eset_1[common_feature_names,]
                    eset_2 <- get(eset_2_name)
                    eset_2 <- eset_2[common_feature_names,]
                    eset_merged <- combine(eset_1, eset_2)
                    if (nrow(dataset_name_combos) > 2) {
                        for (row in 3:nrow(dataset_name_combos)) {
                            eset_n_name <- paste0(c("eset", dataset_name_combos[row,col], suffixes), collapse="_")
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
