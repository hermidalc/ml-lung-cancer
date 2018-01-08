#!/usr/bin/env R

suppressPackageStartupMessages(library("gcrma"))
data <- ReadAffy()
affinities <- compute.affinities("hgu133plus2", verbose=TRUE)
# check.probes("hgu133plus2", "hgu133plus2")
norm_data <- gcrma(data, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
write.table(exprs(norm_data), file="GSE31210_series_matrix_gcrma.txt", sep="\t")
