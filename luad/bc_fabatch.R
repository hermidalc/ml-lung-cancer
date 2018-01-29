#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))

load("data/eset_gex_gse31210_gse8894_gse30219_gse37745.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse30219_gse37745)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse30219_gse37745)
ytr <- as.factor(ptr$Relapse + 1)
btr <- as.factor(ptr$Batch)
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr.Rda")
load("data/eset_gex_gse50081.Rda")
pte <- pData(eset_gex_gse50081)
Xte <- exprs(eset_gex_gse50081)
bte <- as.factor(pte$Batch)
eset_gex_gse50081_fab_te <- eset_gex_gse50081
exprs(eset_gex_gse50081_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse50081_fab_te, file="data/eset_gex_gse50081_fab_te.Rda")
