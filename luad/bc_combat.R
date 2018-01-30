#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    ytr <- as.factor(ptr$Relapse + 1)
    btr <- ptr$Batch
    butr <- sort(unique(btr))
    for (i in 1:length(butr)) {
        if (i != butr[i]) {
            btr <- replace(btr, btr == butr[i], i)
        }
    }
    btr <- as.factor(btr)
    cbt.params <- combatba(t(Xtr), btr)
    eset_tr_cbt <- get(eset_tr_strs[i])
    exprs(eset_tr_cbt) <- t(cbt.params$xadj)
    eset_tr_cbt_str <- paste0(eset_tr_strs[i], "_tr_cbt")
    assign(eset_tr_cbt_str, eset_tr_cbt)
    save(list=eset_tr_cbt_str, file=paste0("data/", eset_tr_cbt_str, ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    ptr <- pData(get(eset_te_strs[i]))
    Xte <- exprs(get(eset_te_strs[i]))
    bte <- pte$Batch
    bute <- sort(unique(bte))
    for (i in 1:length(bute)) {
        if (i != bute[i]) {
            bte <- replace(bte, bte == bute[i], i)
        }
    }
    bte <- as.factor(bte)
    eset_te_cbt <- get(eset_te_strs[i])
    exprs(eset_te_cbt) <- t(combatbaaddon(cbt.params, t(Xte), bte))
    eset_te_cbt_str <- paste0(eset_te_strs[i], "_te_cbt")
    assign(eset_te_cbt_str, eset_te_cbt)
    save(list=eset_te_cbt_str, file=paste0("data/", eset_te_cbt_str, ".Rda"))
}

# pheno <- pData(eset_gex_merged)
# exprs <- exprs(eset_gex_merged)
# batch <- pheno$Batch
# exprs_cbt <- ComBat(dat=exprs, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
# eset_gex_merged_cbt <- eset_gex_merged
# exprs(eset_gex_merged_cbt) <- exprs_cbt
