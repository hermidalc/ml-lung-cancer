#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")

for (i in 1:length(eset_tr_strs)) {
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
    fab_params <- fabatch(t(Xtr), ytr, btr)
    eset_tr_fab <- get(eset_tr_strs[i])
    exprs(eset_tr_fab) <- t(fab_params$xadj)
    eset_tr_fab_str <- paste0(eset_tr_strs[i], "_tr_fab")
    assign(eset_tr_fab_str, eset_tr_fab)
    save(list=eset_tr_fab_str, file=paste0("data/", eset_tr_fab_str, ".Rda"))
    save(fab_params, file=paste0("data/", eset_tr_fab_str, "_params.Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    ptr <- pData(get(eset_te_strs[i]))
    Xte <- exprs(get(eset_te_strs[i]))
    bte <- pte$Batch
    bute <- sort(unique(bte))
    for (j in 1:length(bute)) {
        if (j != bute[j]) {
            bte <- replace(bte, bte == bute[j], j)
        }
    }
    bte <- as.factor(bte)
    eset_te_fab <- get(eset_te_strs[i])
    exprs(eset_te_fab) <- t(fabatchaddon(fab_params, t(Xte), bte))
    eset_te_fab_str <- paste0(eset_te_strs[i], "_te_fab")
    assign(eset_te_fab_str, eset_te_fab)
    save(list=eset_te_fab_str, file=paste0("data/", eset_te_fab_str, ".Rda"))
}
