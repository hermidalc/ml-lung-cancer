#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    eset_tr_cbt_str <- paste0(eset_tr_strs[i], "_tr_cbt")
    eset_te_cbt_str <- paste0(eset_te_strs[i], "_te_cbt")
    print(paste(eset_tr_cbt_str, "->", eset_te_cbt_str))
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    ytr <- as.factor(ptr$Relapse + 1)
    btr <- ptr$Batch
    butr <- sort(unique(btr))
    for (j in 1:length(butr)) {
        if (j != butr[j]) {
            btr <- replace(btr, btr == butr[j], j)
        }
    }
    btr <- as.factor(btr)
    cbt_obj <- combatba(t(Xtr), btr)
    eset_tr_cbt <- get(eset_tr_strs[i])
    exprs(eset_tr_cbt) <- t(cbt_obj$xadj)
    assign(eset_tr_cbt_str, eset_tr_cbt)
    save(list=eset_tr_cbt_str, file=paste0("data/", eset_tr_cbt_str, ".Rda"))
    eset_tr_cbt_obj_str <- paste0(eset_tr_cbt_str, "_obj")
    assign(eset_tr_cbt_obj_str, cbt_obj)
    save(list=eset_tr_cbt_obj_str, file=paste0("data/", eset_tr_cbt_obj_str, ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    pte <- pData(get(eset_te_strs[i]))
    Xte <- exprs(get(eset_te_strs[i]))
    bte <- pte$Batch
    bute <- sort(unique(bte))
    for (j in 1:length(bute)) {
        if (j != bute[j]) {
            bte <- replace(bte, bte == bute[j], j)
        }
    }
    bte <- as.factor(bte)
    eset_te_cbt <- get(eset_te_strs[i])
    exprs(eset_te_cbt) <- t(combatbaaddon(cbt_obj, t(Xte), bte))
    assign(eset_te_cbt_str, eset_te_cbt)
    save(list=eset_te_cbt_str, file=paste0("data/", eset_te_cbt_str, ".Rda"))
}

# exprs_cbt <- ComBat(dat=exprs, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)
