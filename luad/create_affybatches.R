#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
id_type <- cmd_args[1]
if (id_type == "gene") {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
if ("gcrma" %in% cmd_args[2:length(cmd_args)]) {
    cat("Loading CDF:", cdfname, "\n")
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
for (dataset_name in dataset_names) {
    if (dir.exists(paste0("data/raw/", dataset_name))) {
        for (norm_type in cmd_args[2:length(cmd_args)]) {
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
            }
            else if (norm_type == "rma") {
                cat("Performing background correction\n")
                affybatch <- bg.correct.rma(affybatch)
            }
            assign(affybatch_name, affybatch)
            save(list=affybatch_name, file=paste0("data/", affybatch_name, ".Rda"))
        }
    }
}
