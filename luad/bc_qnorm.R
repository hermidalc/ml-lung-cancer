#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")

for (i in 1:length(eset_single_tr_names)) {
    eset_tr_qnorm_name <- paste0(eset_single_tr_names[i], "_tr_qnorm")
    print(eset_tr_qnorm_name)
    load(paste0("data/", eset_single_tr_names[i], ".Rda"))
    Xtr <- t(exprs(get(eset_single_tr_names[i])))
    qnorm_obj <- qunormtrain(Xtr)
    eset_tr_qnorm <- get(eset_single_tr_names[i])
    exprs(eset_tr_qnorm) <- t(qnorm_obj$xnorm)
    assign(eset_tr_qnorm_name, eset_tr_qnorm)
    save(list=eset_tr_qnorm_name, file=paste0("data/", eset_tr_qnorm_name, ".Rda"))
    eset_tr_qnorm_obj_name <- paste0(eset_tr_qnorm_name, "_obj")
    assign(eset_tr_qnorm_obj_name, qnorm_obj)
    save(list=eset_tr_qnorm_obj_name, file=paste0("data/", eset_tr_qnorm_obj_name, ".Rda"))
    for (j in 1:length(eset_single_te_names)) {
        eset_te_qnorm_name <- paste0(eset_single_te_names[j], "_te_qnorm")
        print(paste(eset_tr_qnorm_name, "->", eset_te_qnorm_name))
        load(paste0("data/", eset_single_te_names[j], ".Rda"))
        Xte <- t(exprs(get(eset_single_te_names[j])))
        eset_te_qnorm <- get(eset_single_te_names[j])
        exprs(eset_te_qnorm) <- t(qunormaddon(qnorm_obj, Xte))
        assign(eset_te_qnorm_name, eset_te_qnorm)
        save(list=eset_te_qnorm_name, file=paste0("data/", eset_te_qnorm_name, ".Rda"))
    }
}
