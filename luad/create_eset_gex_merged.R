#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
load("data/eset_gex_gse31210.Rda")
load("data/eset_gex_gse8894.Rda")
load("data/eset_gex_gse30219.Rda")
load("data/eset_gex_gse37745.Rda")
load("data/eset_gex_gse50081.Rda")
# filter common pheno data
pData(eset_gex_gse31210) <- pData(eset_gex_gse31210)[c("Relapse","Gender","Batch")]
pData(eset_gex_gse8894) <- pData(eset_gex_gse8894)[c("Relapse","Gender","Batch")]
pData(eset_gex_gse30219) <- pData(eset_gex_gse30219)[c("Relapse","Gender","Batch")]
pData(eset_gex_gse37745) <- pData(eset_gex_gse37745)[c("Relapse","Gender","Batch")]
pData(eset_gex_gse50081) <- pData(eset_gex_gse50081)[c("Relapse","Gender","Batch")]
# merge
eset_gex_gse31210_gse8894_gse30219_gse37745 <- combine(eset_gex_gse31210, eset_gex_gse8894)
eset_gex_gse31210_gse8894_gse30219_gse37745 <- combine(eset_gex_gse31210_gse8894_gse30219_gse37745, eset_gex_gse30219)
eset_gex_gse31210_gse8894_gse30219_gse37745 <- combine(eset_gex_gse31210_gse8894_gse30219_gse37745, eset_gex_gse37745)

eset_gex_gse31210_gse8894_gse30219_gse50081 <- combine(eset_gex_gse31210, eset_gex_gse8894)
eset_gex_gse31210_gse8894_gse30219_gse50081 <- combine(eset_gex_gse31210_gse8894_gse30219_gse50081, eset_gex_gse30219)
eset_gex_gse31210_gse8894_gse30219_gse50081 <- combine(eset_gex_gse31210_gse8894_gse30219_gse50081, eset_gex_gse50081)

eset_gex_gse31210_gse8894_gse37745_gse50081 <- combine(eset_gex_gse31210, eset_gex_gse8894)
eset_gex_gse31210_gse8894_gse37745_gse50081 <- combine(eset_gex_gse31210_gse8894_gse37745_gse50081, eset_gex_gse37745)
eset_gex_gse31210_gse8894_gse37745_gse50081 <- combine(eset_gex_gse31210_gse8894_gse37745_gse50081, eset_gex_gse50081)

eset_gex_gse31210_gse30219_gse37745_gse50081 <- combine(eset_gex_gse31210, eset_gex_gse30219)
eset_gex_gse31210_gse30219_gse37745_gse50081 <- combine(eset_gex_gse31210_gse30219_gse37745_gse50081, eset_gex_gse37745)
eset_gex_gse31210_gse30219_gse37745_gse50081 <- combine(eset_gex_gse31210_gse30219_gse37745_gse50081, eset_gex_gse50081)

eset_gex_gse8894_gse30219_gse37745_gse50081 <- combine(eset_gex_gse8894, eset_gex_gse30219)
eset_gex_gse8894_gse30219_gse37745_gse50081 <- combine(eset_gex_gse8894_gse30219_gse37745_gse50081, eset_gex_gse37745)
eset_gex_gse8894_gse30219_gse37745_gse50081 <- combine(eset_gex_gse8894_gse30219_gse37745_gse50081, eset_gex_gse50081)

# save
save(eset_gex_gse31210_gse8894_gse30219_gse37745, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse50081, file="data/eset_gex_gse31210_gse8894_gse30219_gse50081.Rda")
save(eset_gex_gse31210_gse8894_gse37745_gse50081, file="data/eset_gex_gse31210_gse8894_gse37745_gse50081.Rda")
save(eset_gex_gse31210_gse30219_gse37745_gse50081, file="data/eset_gex_gse31210_gse30219_gse37745_gse50081.Rda")
save(eset_gex_gse8894_gse30219_gse37745_gse50081, file="data/eset_gex_gse8894_gse30219_gse37745_gse50081.Rda")
