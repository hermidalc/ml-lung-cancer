#!/usr/bin/env R

suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
data <- ReadAffy()
affinities <- compute.affinities("hgu133plus2", verbose=TRUE)
check.probes("hgu133plus2", "hgu133plus2")
norm_data <- gcrma(data, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
write.table(exprs(norm_data), file="GSE50081_series_matrix.txt", sep="\t")
save(norm_data, file="norm_data_gcrma_gse50081.Rda")
