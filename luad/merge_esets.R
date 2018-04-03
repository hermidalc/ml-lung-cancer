#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_subset <- as.integer(cmd_args[1])
norm_type <- cmd_args[2]
id_type <- cmd_args[3]
for (dataset_name in dataset_names) {
    if (is.na(id_type) | id_type == "none") {
        eset_name <- paste0(c("eset", dataset_name, norm_type), collapse="_")
    }
    else {
        eset_name <- paste0(c("eset", dataset_name, norm_type, id_type), collapse="_")
    }
    eset_file <- paste0("data/", eset_name, ".Rda")
    if (file.exists(eset_file)) {
        print(paste("Loading:", eset_name), quote=FALSE)
        load(eset_file)
        # subset common pheno data
        eset <- get(eset_name)
        pData(eset) <- pData(eset)[common_pheno_colnames]
        assign(eset_name, eset)
    }
}
if (is.na(id_type) || id_type == "none") {
    dataset_name_combos <- combn(dataset_names, num_subset, FUN=paste, simplify=TRUE, norm_type, sep="_")
} else {
    dataset_name_combos <- combn(dataset_names, num_subset, FUN=paste, simplify=TRUE, norm_type, id_type, sep="_")
}
for (col in 1:ncol(dataset_name_combos)) {
    eset_merged_name <- paste0(c("eset", dataset_name_combos[,col]), collapse="_")
    print(paste("Creating:", eset_merged_name), quote=FALSE)
    eset_1 <- get(paste0(c("eset", dataset_name_combos[1,col]), collapse="_"))
    eset_2 <- get(paste0(c("eset", dataset_name_combos[2,col]), collapse="_"))
    eset_merged <- combine(eset_1, eset_2)
    if (nrow(dataset_name_combos) > 2) {
        for (row in 3:nrow(dataset_name_combos)) {
            eset_n <- get(paste0(c("eset", dataset_name_combos[row,col]), collapse="_"))
            eset_merged <- combine(eset_merged, eset_n)
        }
    }
    assign(eset_merged_name, eset_merged)
    save(list=eset_merged_name, file=paste0("data/", eset_merged_name, ".Rda"))
}
