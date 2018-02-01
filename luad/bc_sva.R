#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    eset_tr_sva_str <- paste0(eset_tr_strs[i], "_tr_sva")
    eset_te_sva_str <- paste0(eset_te_strs[i], "_te_sva")
    print(paste(eset_tr_sva_str, "->", eset_te_sva_str))
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
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
    # ctrls <- as.numeric(grepl("^AFFX", rownames(Xtr)))
    sva_params <- svaba(t(Xtr), btr, mod, mod0, algorithm="fast")
    eset_tr_sva <- get(eset_tr_strs[i])
    exprs(eset_tr_sva) <- t(sva_params$xadj)
    assign(eset_tr_sva_str, eset_tr_sva)
    save(list=eset_tr_sva_str, file=paste0("data/", eset_tr_sva_str, ".Rda"))
    eset_tr_sva_params_str <- paste0(eset_tr_sva_str, "_params")
    assign(eset_tr_sva_params_str, sva_params)
    save(list=eset_tr_sva_params_str, file=paste0("data/", eset_tr_sva_params_str, ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    Xte <- exprs(get(eset_te_strs[i]))
    eset_te_sva <- get(eset_te_strs[i])
    exprs(eset_te_sva) <- t(svabaaddon(sva_params, t(Xte)))
    assign(eset_te_sva_str, eset_te_sva)
    save(list=eset_te_sva_str, file=paste0("data/", eset_te_sva_str, ".Rda"))
}
