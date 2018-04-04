#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_subset <- as.integer(cmd_args[1])
norm_type <- cmd_args[2]
id_type <- cmd_args[3]
suffixes <- c(norm_type)
if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
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
dataset_name_combos <- combn(dataset_names, num_subset)
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
    }
}
