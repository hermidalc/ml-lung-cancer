#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("sva"))
load("data/eset_gex_merged.Rda")
pheno <- pData(eset_gex_merged)
exprs <- exprs(eset_gex_merged)
batch <- pheno$Batch
exprs_bc <- ComBat(dat=exprs, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
eset_gex_merged_bc <- eset_gex_merged
exprs(eset_gex_merged_bc) <- exprs_bc
eset_gex_gse31210_bc <- eset_gex_merged_bc[, eset_gex_merged_bc$Batch == 1]
eset_gex_gse30219_bc <- eset_gex_merged_bc[, eset_gex_merged_bc$Batch == 2]
eset_gex_gse37745_bc <- eset_gex_merged_bc[, eset_gex_merged_bc$Batch == 3]
eset_gex_gse50081_bc <- eset_gex_merged_bc[, eset_gex_merged_bc$Batch == 4]
save(eset_gex_gse31210_bc, file="data/eset_gex_gse31210_bc.Rda")
save(eset_gex_gse30219_bc, file="data/eset_gex_gse30219_bc.Rda")
save(eset_gex_gse37745_bc, file="data/eset_gex_gse37745_bc.Rda")
save(eset_gex_gse50081_bc, file="data/eset_gex_gse50081_bc.Rda")
