#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("readxl"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("impute"))
suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(library("bapred"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("lib/R/gcrmapred.R")
source("lib/R/config.R")

# cmd_args <- commandArgs(trailingOnly=TRUE)
# affybatch <- ReadAffy(celfile.path=cmd_args[2], cdfname="hgu133plus2", verbose=TRUE)
# write.table(exprs(eset), file=paste0(cmd_args[1], "_series_matrix.txt"), sep="\t")
# eset_name <- paste0(c("eset", cmd_args[1], "gcrma"), collapse="_")
# assign(eset_name, eset)
# save(list=eset_name, file=paste0(eset_name, ".Rda"))

cmd_args <- commandArgs(trailingOnly=TRUE)
num_subset <- cmd_args[1]
dataset_names <- dataset_names[1:num_subset]
dataset_name_combos <- combn(dataset_names, length(dataset_names) - 1)
for (norm_type in cmd_args[2:length(cmd_args)]) {
    if (norm_type %in% c("gcrma", "rma")) {
        for (col in 1:ncol(dataset_name_combos)) {
            cel_files <- c()
            for (dataset_name in dataset_name_combos[,col]) {
                cel_path <- paste0("data/raw/", dataset_name)
                if (dir.exists(cel_path)) {
                    cel_files <- append(cel_files, list.files(path=cel_path, full.names=TRUE, pattern="\\.CEL$"))
                }
                else {
                    cel_files <- NULL
                    break
                }
            }
            if (!is.null(cel_files)) {
                eset_norm_name <- paste0(c("eset", dataset_name_combos[,col], "gcrma"), collapse="_")
                print(paste("Creating: ", eset_norm_name))
                affybatch <- ReadAffy(filenames=cel_files, cdfname="hgu133plus2", verbose=TRUE)
                if (norm_type == "gcrma") {
                    norm_obj <- gcrmatrain(affybatch)
                }
                else if (norm_type == "rma") {
                    norm_obj <- rmatrain(affybatch)
                }
                eset_norm <- ExpressionSet(assayData=t(norm_obj$xnorm), phenoData=pheno)
            }
        }
    }
}
