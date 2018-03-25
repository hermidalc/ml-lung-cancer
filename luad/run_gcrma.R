#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))

cmd_args <- commandArgs(trailingOnly=TRUE)
affybatch <- ReadAffy(celfile.path=cmd_args[2], cdfname="hgu133plus2", verbose=TRUE)
affinities <- compute.affinities("hgu133plus2", verbose=TRUE)
# fails with strange error
#check.probes("hgu133plus2", "hgu133plus2")
eset <- gcrma(affybatch, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
write.table(exprs(eset), file=paste0(cmd_args[1], "_series_matrix.txt"), sep="\t")
eset_name <- paste0(c("eset", cmd_args[1], "gcrma"), collapse="_")
assign(eset_name, eset)
save(list=eset_name, file=paste0(eset_name, ".Rda"))
