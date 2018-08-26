#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--filter", type="character", nargs="+", help="filter function")
parser$add_argument("--datasets-tr", type="character", nargs="+", help="datasets tr")
parser$add_argument("--datasets-te", type="character", nargs="+", help="datasets te")
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument("--prep-meth", type="character", nargs="+", help="preprocess method")
parser$add_argument("--bc-meth", type="character", nargs="+", help="batch correction method")
parser$add_argument("--feat-file", type="character", nargs=1, help="feature file")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
args <- parser$parse_args()
if (is.null(args$filter) || !(args$filter %in% c("common_features", "features"))) {
    stop("--filter option required")
}
if (args$filter == "features") {
    if (is.null(args$num_tr_combo)) stop("--num-tr-combo option required")
    if (is.null(args$feat_file)) stop("--feat-file option required")
    if (!file.exists(args$feat_file)) stop("Invalid feature file")
}
if (!is.null(args$datasets_tr) && !is.null(args$num_tr_combo)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$datasets_tr), as.integer(args$num_tr_combo))
} else if (!is.null(args$datasets_tr)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$datasets_tr), length(args$datasets_tr))
} else {
    dataset_tr_name_combos <- combn(dataset_names, as.integer(args$num_tr_combo))
}
if (!is.null(args$datasets_tr)) {
    dataset_names <- intersect(dataset_names, args$datasets_tr)
}
if (!is.null(args$datasets_te)) {
    dataset_te_names <- intersect(dataset_names, args$datasets_te)
} else {
    dataset_te_names <- dataset_names
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
if (!is.null(args$prep_meth)) {
    prep_methods <- intersect(prep_methods, args$prep_meth)
}
if (!is.null(args$bc_meth)) {
    bc_methods <- intersect(bc_methods, args$bc_meth)
}
if (args$filter == "common_features") {
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
                        cat(" Loading:", eset_name, "\n")
                        load(eset_file)
                        # get common features
                        if (args$filter == "common_features") {
                            if (exists("common_feature_names")) {
                                common_feature_names <- intersect(
                                    common_feature_names, featureNames(get(eset_name))
                                )
                            }
                            else {
                                common_feature_names <- featureNames(get(eset_name))
                            }
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
                        eset_flt_name <- paste0(c(eset_name, "cff"), collapse="_")
                        cat("Creating:", eset_flt_name, "\n")
                        eset_flt <- get(eset_name)
                        eset_flt <- eset_flt[common_feature_names,]
                        assign(eset_flt_name, eset_flt)
                        save(list=eset_flt_name, file=paste0("data/", eset_flt_name, ".Rda"))
                    }
                }
            }
        }
    }
} else if (args$filter == "features") {
    cat("Reading ", basename(args$feat_file), ": ", sep="")
    feature_names <- unique(readLines(args$feat_file))
    cat(length(feature_names), "unique features\n")
    filter_type <- strsplit(basename(args$feat_file), split="_", fixed=TRUE)[[1]][1]
    for (col in 1:ncol(dataset_tr_name_combos)) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    for (prep_meth in prep_methods) {
                        for (bc_meth in bc_methods) {
                            suffixes_tr <- suffixes
                            suffixes_te <- suffixes
                            if (prep_meth != "none") {
                                suffixes_tr <- c(suffixes_tr, prep_meth)
                                if (bc_meth != "none") {
                                    suffixes_tr <- c(suffixes_tr, bc_meth)
                                }
                                if (prep_meth != "mrg") {
                                    suffixes_te <- suffixes_tr
                                }
                                else if (bc_meth != "none") {
                                    suffixes_te <- c(suffixes_te, bc_meth)
                                }
                            }
                            else if (bc_meth != "none") {
                                suffixes_tr <- c(suffixes_tr, bc_meth)
                                suffixes_te <- suffixes_tr
                            }
                            if (length(dataset_tr_name_combos[,col]) > 1 || bc_meth != "none") {
                                eset_tr_name <- paste0(
                                    c("eset", dataset_tr_name_combos[,col], suffixes_tr, "tr"),
                                    collapse="_"
                                )
                                eset_tr_flt_name <- paste0(
                                    c("eset", dataset_tr_name_combos[,col], suffixes_tr, filter_type, "tr"),
                                    collapse="_"
                                )
                            }
                            else {
                                eset_tr_name <- paste0(
                                    c("eset", dataset_tr_name_combos[,col], suffixes_tr), collapse="_"
                                )
                                eset_tr_flt_name <- paste0(c(eset_tr_name, filter_type), collapse="_")
                            }
                            eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
                            if (!exists(eset_tr_name)) {
                                if (file.exists(eset_tr_file)) {
                                    cat(" Loading:", eset_tr_name, "\n")
                                    load(eset_tr_file)
                                }
                                else {
                                    next
                                }
                                if (args$load_only) next
                                cat("Creating:", eset_tr_flt_name, "\n")
                                eset_tr_flt <- get(eset_tr_name)
                                eset_tr_flt <- eset_tr_flt[feature_names,]
                                assign(eset_tr_flt_name, eset_tr_flt)
                                save(list=eset_tr_flt_name, file=paste0("data/", eset_tr_flt_name, ".Rda"))
                                remove(list=c(eset_tr_flt_name))
                            }
                            for (dataset_te_name in setdiff(dataset_te_names, dataset_tr_name_combos[,col])) {
                                if (length(dataset_tr_name_combos[,col]) > 1 || bc_meth != "none") {
                                    eset_te_name <- paste0(
                                        c(eset_tr_name, dataset_te_name, "te"), collapse="_"
                                    )
                                    eset_te_flt_name <- paste0(
                                        c(eset_tr_flt_name, dataset_te_name, "te"), collapse="_"
                                    )
                                }
                                else {
                                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes_te), collapse="_")
                                    eset_te_flt_name <- paste0(c(eset_te_name, filter_type), collapse="_")
                                }
                                eset_te_file <- paste0("data/", eset_te_name, ".Rda")
                                if (!exists(eset_te_name)) {
                                    if (file.exists(eset_te_file)) {
                                        cat(" Loading:", eset_te_name, "\n")
                                        load(eset_te_file)
                                    }
                                    else {
                                        next
                                    }
                                    if (args$load_only) next
                                    cat("Creating:", eset_te_flt_name, "\n")
                                    eset_te_flt <- get(eset_te_name)
                                    eset_te_flt <- eset_te_flt[feature_names,]
                                    assign(eset_te_flt_name, eset_te_flt)
                                    save(list=eset_te_flt_name, file=paste0("data/", eset_te_flt_name, ".Rda"))
                                    remove(list=c(eset_te_flt_name))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}