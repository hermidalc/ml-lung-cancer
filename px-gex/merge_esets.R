#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--norm-meth", type="character", nargs="+", help="preprocessing/normalization method")
parser$add_argument("--id-type", type="character", nargs="+", help="dataset id type")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
args <- parser$parse_args()

num_tr_combo <- as.integer(args$num_tr_combo)
if (!is.null(args$norm_meth)) {
    norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
}
if (!is.null(args$id_type)) {
    id_types <- id_types[id_types %in% args$id_type]
}
for (norm_meth in norm_methods) {
    for (id_type in id_types) {
        suffixes <- c(norm_meth)
        if (id_type != "none") suffixes <- c(suffixes, id_type)
        for (dataset_name in dataset_names) {
            eset_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
            eset_file <- paste0("data/", eset_name, ".Rda")
            if (file.exists(eset_file)) {
                cat("Loading:", eset_name, "\n")
                load(eset_file)
                # subset common pheno data
                eset <- get(eset_name)
                pData(eset) <- pData(eset)[common_pheno_colnames]
                assign(eset_name, eset)
            }
        }
        if (args$load_only) next
        dataset_name_combos <- combn(dataset_names, num_tr_combo)
        for (col in 1:ncol(dataset_name_combos)) {
            eset_merged_name <- paste0(c("eset", dataset_name_combos[,col], suffixes, "merged", "tr"), collapse="_")
            eset_1_name <- paste0(c("eset", dataset_name_combos[1,col], suffixes), collapse="_")
            eset_2_name <- paste0(c("eset", dataset_name_combos[2,col], suffixes), collapse="_")
            if (exists(eset_1_name) & exists(eset_2_name)) {
                cat("Creating:", eset_merged_name, "\n")
                eset_merged <- combine(get(eset_1_name), get(eset_2_name))
                if (nrow(dataset_name_combos) > 2) {
                    for (row in 3:nrow(dataset_name_combos)) {
                        eset_n_name <- paste0(c("eset", dataset_name_combos[row,col], suffixes), collapse="_")
                        eset_merged <- combine(eset_merged, get(eset_n_name))
                    }
                }
                assign(eset_merged_name, eset_merged)
                save(list=eset_merged_name, file=paste0("data/", eset_merged_name, ".Rda"))
                remove(list=c(eset_merged_name))
            }
        }
    }
}
