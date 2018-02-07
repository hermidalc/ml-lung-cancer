#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")

for (i in 1:length(eset_merged_tr_names)) {
    eset_tr_std_name <- paste0(eset_merged_tr_names[i], "_tr_std")
    eset_te_std_name <- paste0(eset_merged_te_names[i], "_te_std")
    print(paste(eset_tr_std_name, "->", eset_te_std_name))
    load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
    ptr <- pData(get(eset_merged_tr_names[i]))
    Xtr <- t(exprs(get(eset_merged_tr_names[i])))
    ytr <- as.factor(ptr$Relapse + 1)
    btr <- ptr$Batch
    butr <- sort(unique(btr))
    for (j in 1:length(butr)) {
        if (j != butr[j]) {
            btr <- replace(btr, btr == butr[j], j)
        }
    }
    btr <- as.factor(btr)
    std_obj <- standardize(Xtr, btr)
    eset_tr_std <- get(eset_merged_tr_names[i])
    exprs(eset_tr_std) <- t(std_obj$xadj)
    assign(eset_tr_std_name, eset_tr_std)
    save(list=eset_tr_std_name, file=paste0("data/", eset_tr_std_name, ".Rda"))
    eset_tr_std_obj_name <- paste0(eset_tr_std_name, "_obj")
    assign(eset_tr_std_obj_name, std_obj)
    save(list=eset_tr_std_obj_name, file=paste0("data/", eset_tr_std_obj_name, ".Rda"))
    load(paste0("data/", eset_merged_te_names[i], ".Rda"))
    pte <- pData(get(eset_merged_te_names[i]))
    Xte <- t(exprs(get(eset_merged_te_names[i])))
    bte <- pte$Batch
    bute <- sort(unique(bte))
    for (j in 1:length(bute)) {
        if (j != bute[j]) {
            bte <- replace(bte, bte == bute[j], j)
        }
    }
    bte <- as.factor(bte)
    eset_te_std <- get(eset_merged_te_names[i])
    exprs(eset_te_std) <- t(standardizeaddon(std_obj, Xte, bte))
    assign(eset_te_std_name, eset_te_std)
    save(list=eset_te_std_name, file=paste0("data/", eset_te_std_name, ".Rda"))
}
