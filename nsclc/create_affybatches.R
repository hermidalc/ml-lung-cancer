#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
batch_type <- cmd_args[1]
num_subset <- cmd_args[2]
id_type <- cmd_args[3]
if (id_type == "gene") {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
if ("gcrma" %in% cmd_args[4:length(cmd_args)]) {
    cat("Loading CDF:", cdfname, "\n")
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
if (batch_type == "single") {
    for (dataset_name in dataset_names) {
        if (!dir.exists(paste0("data/raw/", dataset_name))) next
        for (norm_type in cmd_args[4:length(cmd_args)]) {
            suffixes <- c(norm_type)
            if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
            affybatch_name <- paste0(c("affybatch", dataset_name, suffixes), collapse="_")
            cat("Creating AffyBatch:", affybatch_name, "\n")
            affybatch <- ReadAffy(
                celfile.path=paste0("data/raw/", dataset_name), cdfname=cdfname, verbose=TRUE
            )
            if (norm_type == "gcrma") {
                affybatch <- bg.adjust.gcrma(
                    affybatch, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
                )
            } else if (norm_type == "rma") {
                cat("Performing background correction\n")
                affybatch <- bg.correct.rma(affybatch)
            }
            assign(affybatch_name, affybatch)
            save(list=affybatch_name, file=paste0("data/", affybatch_name, ".Rda"))
        }
    }
# combo only needed for gcrma currently
} else if (batch_type == "combo" & "gcrma" %in% cmd_args[4:length(cmd_args)]) {
    dataset_name_combos <- combn(dataset_names, num_subset)
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
        if (is.null(cel_files)) next
        for (norm_type in cmd_args[4:length(cmd_args)]) {
            suffixes <- c(norm_type)
            if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
            affybatch_name <- paste0(c("affybatch", dataset_name_combos[,col], suffixes), collapse="_")
            cat("Creating AffyBatch:", affybatch_name, "\n")
            affybatch <- ReadAffy(filenames=cel_files, cdfname=cdfname, verbose=TRUE)
            affybatch <- bg.adjust.gcrma(
                affybatch, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
            )
            assign(affybatch_name, affybatch)
            save(list=affybatch_name, file=paste0("data/", affybatch_name, ".Rda"))
        }
    }
}
