#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("genefilter"))
suppressPackageStartupMessages(library("sva"))

source("normFact.R")
load("data/eset_gex_merged.Rda")
pheno <- pData(eset_gex_merged)
exprs <- exprs(eset_gex_merged)
batch <- pheno$Batch
icaobj <- normFact("stICA", exprs, batch, "categorical", k=20, alpha=0.5)
exprs_ica <- icaobj$Xn
eset_gex_merged_ica <- eset_gex_merged
exprs(eset_gex_merged_ica) <- exprs_ica
# filter out control probesets
eset_gex_merged_ica <- featureFilter(eset_gex_merged_ica,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
eset_gex_gse31210_ica <- eset_gex_merged_ica[, eset_gex_merged_ica$Batch == 1]
eset_gex_gse30219_ica <- eset_gex_merged_ica[, eset_gex_merged_ica$Batch == 2]
eset_gex_gse37745_ica <- eset_gex_merged_ica[, eset_gex_merged_ica$Batch == 3]
eset_gex_gse50081_ica <- eset_gex_merged_ica[, eset_gex_merged_ica$Batch == 4]
save(eset_gex_merged_ica, file="data/eset_gex_merged_ica.Rda")
save(eset_gex_gse31210_ica, file="data/eset_gex_gse31210_ica.Rda")
save(eset_gex_gse30219_ica, file="data/eset_gex_gse30219_ica.Rda")
save(eset_gex_gse37745_ica, file="data/eset_gex_gse37745_ica.Rda")
save(eset_gex_gse50081_ica, file="data/eset_gex_gse50081_ica.Rda")
