#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
load("data/eset_gex_gse31210.Rda")
pData(eset_gex_gse31210) <- pData(eset_gex_gse31210)[c("Relapse","Sex","Batch")]
pData(eset_gex_gse8894) <- pData(eset_gex_gse8894)[c("Relapse","Sex","Batch")]
pData(eset_gex_gse30219) <- pData(eset_gex_gse30219)[c("Relapse","Sex","Batch")]
pData(eset_gex_gse37745) <- pData(eset_gex_gse37745)[c("Relapse","Sex","Batch")]
pData(eset_gex_gse50081) <- pData(eset_gex_gse50081)[c("Relapse","Sex","Batch")]
# merge
eset_gex_merged <- combine(eset_gex_gse31210, eset_gex_gse8894)
eset_gex_merged <- combine(eset_gex_merged, eset_gex_gse30219)
eset_gex_merged <- combine(eset_gex_merged, eset_gex_gse37745)
eset_gex_merged <- combine(eset_gex_merged, eset_gex_gse50081)
# save
save(eset_gex_merged, file="data/eset_gex_merged.Rda")
