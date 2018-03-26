#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
source("lib/R/svaba.R")
source("lib/R/config.R")

# cmd_args <- commandArgs(trailingOnly=TRUE)
# affybatch <- ReadAffy(celfile.path=cmd_args[2], cdfname="hgu133plus2", verbose=TRUE)
# write.table(exprs(eset), file=paste0(cmd_args[1], "_series_matrix.txt"), sep="\t")
# eset_name <- paste0(c("eset", cmd_args[1], "gcrma"), collapse="_")
# assign(eset_name, eset)
# save(list=eset_name, file=paste0(eset_name, ".Rda"))

cmd_args <- commandArgs(trailingOnly=TRUE)
dataset_name_combos <- combn(dataset_names, length(dataset_names) - 1)
for (norm_type in cmd_args) {
    if (norm_type %in% c("rma", "gcrma")) {
        for (col in 1:ncol(dataset_name_combos)) {
            print(dataset_name_combos[,col])
        }
    }
}
