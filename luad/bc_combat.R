#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
suppressPackageStartupMessages(library("bapred"))

load("data/eset_gex_gse31210_gse8894_gse30219_gse37745.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse30219_gse37745)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse30219_gse37745)
btr <- as.factor(ptr$Batch)
cbt.params <- combatba(t(Xtr), btr)
eset_gex_gse31210_gse8894_gse30219_gse37745_cbt_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_cbt_tr) <- t(cbt.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse37745_cbt_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_cbt_tr.Rda")
load("data/eset_gex_gse50081.Rda")
pte <- pData(eset_gex_gse50081)
Xte <- exprs(eset_gex_gse50081)
bte <- as.factor(pte$Batch)
eset_gex_gse50081_cbt_te <- eset_gex_gse50081
exprs(eset_gex_gse50081_cbt_te) <- t(combatbaaddon(cbt.params, t(Xte), bte))
save(eset_gex_gse50081_cbt_te, file="data/eset_gex_gse50081_cbt_te.Rda")

# pheno <- pData(eset_gex_merged)
# exprs <- exprs(eset_gex_merged)
# batch <- pheno$Batch
# exprs_cbt <- ComBat(dat=exprs, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
# eset_gex_merged_cbt <- eset_gex_merged
# exprs(eset_gex_merged_cbt) <- exprs_cbt
