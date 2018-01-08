#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("genefilter"))
suppressPackageStartupMessages(library("sva"))
load("data/eset_gex_merged.Rda")
pheno <- pData(eset_gex_merged)
exprs <- exprs(eset_gex_merged)
batch <- pheno$Batch
exprs_cbt <- ComBat(dat=exprs, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
eset_gex_merged_cbt <- eset_gex_merged
exprs(eset_gex_merged_cbt) <- exprs_cbt
# filter out control probesets
eset_gex_merged_cbt <- featureFilter(eset_gex_merged_cbt,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
eset_gex_gse31210_cbt <- eset_gex_merged_cbt[, eset_gex_merged_cbt$Batch == 1]
eset_gex_gse30219_cbt <- eset_gex_merged_cbt[, eset_gex_merged_cbt$Batch == 2]
eset_gex_gse37745_cbt <- eset_gex_merged_cbt[, eset_gex_merged_cbt$Batch == 3]
eset_gex_gse50081_cbt <- eset_gex_merged_cbt[, eset_gex_merged_cbt$Batch == 4]
save(eset_gex_merged_cbt, file="data/eset_gex_merged_cbt.Rda")
save(eset_gex_gse31210_cbt, file="data/eset_gex_gse31210_cbt.Rda")
save(eset_gex_gse30219_cbt, file="data/eset_gex_gse30219_cbt.Rda")
save(eset_gex_gse37745_cbt, file="data/eset_gex_gse37745_cbt.Rda")
save(eset_gex_gse50081_cbt, file="data/eset_gex_gse50081_cbt.Rda")
