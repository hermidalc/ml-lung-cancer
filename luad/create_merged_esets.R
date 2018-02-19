#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
source("lib/R/config.R")

for (dataset_name in dataset_names) {
    eset_name <- paste0(c("eset", dataset_name), collapse="_")
    print(paste("Loading:", eset_name))
    load(paste0("data/", eset_name, ".Rda"))
    # subset common pheno data
    eset <- get(eset_name)
    pData(eset) <- pData(eset)[c("Relapse","Gender","Batch")]
    assign(eset_name, eset)
}
# merge (leaving one out each time)
dataset_name_combos <- combn(dataset_names, length(dataset_names) - 1)
for (col in 1:ncol(dataset_name_combos)) {
    eset_merged_name <- paste0(c("eset", dataset_name_combos[,col]), collapse="_")
    print(paste("Creating:", eset_merged_name))
    eset_1 <- get(paste0(c("eset", dataset_name_combos[1,col]), collapse="_"))
    eset_2 <- get(paste0(c("eset", dataset_name_combos[2,col]), collapse="_"))
    eset_merged <- combine(eset_1, eset_2)
    for (row in 3:nrow(dataset_name_combos)) {
        eset_n <- get(paste0(c("eset", dataset_name_combos[row,col]), collapse="_"))
        eset_merged <- combine(eset_merged, eset_n)
    }
    assign(eset_merged_name, eset_merged)
    save(list=eset_merged_name, file=paste0("data/", eset_merged_name, ".Rda"))
}
