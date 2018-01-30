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
    for (i in 1:length(butr)) {
        if (i != butr[i]) {
            btr <- replace(btr, btr == butr[i], i)
        }
    }
    btr <- as.factor(btr)
    fab.params <- fabatch(t(Xtr), ytr, btr)
    eset_tr_fab <- get(eset_tr_strs[i])
    exprs(eset_tr_fab) <- t(fab.params$xadj)
    eset_tr_fab_str <- paste0(eset_tr_strs[i], "_tr_fab")
    assign(eset_tr_fab_str, eset_tr_fab)
    save(get(eset_tr_fab_str), file=paste0("data/", eset_tr_fab_str, ".Rda"))
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
    eset_te_fab <- get(eset_te_strs[i])
    exprs(eset_te_fab) <- t(fabatchaddon(fab.params, t(Xte), bte))
    eset_te_fab_str <- paste0(eset_te_strs[i], "_te_fab")
    assign(eset_te_fab_str, eset_te_fab)
    save(get(eset_te_fab_str), file=paste0("data/", eset_te_fab_str, ".Rda"))
}
