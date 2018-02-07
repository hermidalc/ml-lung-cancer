#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
source("config.R")

for (i in 1:length(eset_merged_tr_names)) {
    eset_tr_sva_name <- paste0(eset_merged_tr_names[i], "_tr_sva")
    eset_te_sva_name <- paste0(eset_merged_te_names[i], "_te_sva")
    print(paste(eset_tr_sva_name, "->", eset_te_sva_name))
    load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
    ptr <- pData(get(eset_merged_tr_names[i]))
    Xtr <- t(exprs(get(eset_merged_tr_names[i])))
    btr <- ptr$Batch
    butr <- sort(unique(btr))
    for (j in 1:length(butr)) {
        if (j != butr[j]) {
            btr <- replace(btr, btr == butr[j], j)
        }
    }
    btr <- as.factor(btr)
    mod <- model.matrix(~as.factor(Relapse), data=ptr)
    mod0 <- model.matrix(~1, data=ptr)
    # ctrls <- as.numeric(grepl("^AFFX", rownames(t(Xtr))))
    sva_obj <- svaba(Xtr, btr, mod, mod0, algorithm="fast")
    eset_tr_sva <- get(eset_merged_tr_names[i])
    exprs(eset_tr_sva) <- t(sva_obj$xadj)
    assign(eset_tr_sva_name, eset_tr_sva)
    save(list=eset_tr_sva_name, file=paste0("data/", eset_tr_sva_name, ".Rda"))
    eset_tr_sva_obj_name <- paste0(eset_tr_sva_name, "_obj")
    assign(eset_tr_sva_obj_name, sva_obj)
    save(list=eset_tr_sva_obj_name, file=paste0("data/", eset_tr_sva_obj_name, ".Rda"))
    load(paste0("data/", eset_merged_te_names[i], ".Rda"))
    Xte <- t(exprs(get(eset_merged_te_names[i])))
    eset_te_sva <- get(eset_merged_te_names[i])
    exprs(eset_te_sva) <- t(svabaaddon(sva_obj, Xte))
    assign(eset_te_sva_name, eset_te_sva)
    save(list=eset_te_sva_name, file=paste0("data/", eset_te_sva_name, ".Rda"))
}
