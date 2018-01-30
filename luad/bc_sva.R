#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    btr <- ptr$Batch
    butr <- sort(unique(btr))
    for (i in 1:length(butr)) {
        if (i != butr[i]) {
            btr <- replace(btr, btr == butr[i], i)
        }
    }
    btr <- as.factor(btr)
    mod <- model.matrix(~as.factor(Relapse), data=ptr)
    mod0 <- model.matrix(~1, data=ptr)
    ctrls <- grepl("^AFFX", rownames(Xtr))
    numsv <- sva::num.sv(Xtr, mod, method="be")
    sva_params <- svaba(t(Xtr), btr, mod, mod0, numsv, controls=ctrls, algorithm="exact")
    eset_tr_sva <- get(eset_tr_strs[i])
    exprs(eset_tr_sva) <- t(sva_params$xadj)
    eset_tr_sva_str <- paste0(eset_tr_strs[i], "_tr_sva")
    assign(eset_tr_sva_str, eset_tr_sva)
    save(list=eset_tr_sva_str, file=paste0("data/", eset_tr_sva_str, ".Rda"))
    save(list=sva_params, file=paste0("data/", eset_tr_sva_str, "_params.Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    Xte <- exprs(get(eset_te_strs[i]))
    eset_te_sva <- get(eset_te_strs[i])
    exprs(eset_te_sva) <- t(svabaaddon(sva_params, t(Xte)))
    eset_te_sva_str <- paste0(eset_te_strs[i], "_te_sva")
    assign(eset_te_sva_str, eset_te_sva)
    save(list=eset_te_sva_str, file=paste0("data/", eset_te_sva_str, ".Rda"))
    # sva_params <- svaba(t(Xtr), btr, mod, mod0, numsv, algorithm="exact")
    # eset_tr_sva <- get(eset_tr_strs[i])
    # exprs(eset_tr_sva) <- t(sva_params$xadj)
    # eset_tr_sva_str <- paste0(eset_tr_strs[i], "_tr_sva")
    # assign(eset_tr_sva_str, eset_tr_sva)
    # save(list=eset_tr_sva_str, file=paste0("data/", eset_tr_sva_str, ".Rda"))
    # eset_te_sva <- get(eset_te_strs[i])
    # exprs(eset_te_sva) <- t(svabaaddon(sva_params, t(Xte)))
    # eset_te_sva_str <- paste0(eset_te_strs[i], "_te_sva")
    # assign(eset_te_sva_str, eset_te_sva)
    # save(list=eset_te_sva_str, file=paste0("data/", eset_te_sva_str, ".Rda"))
}

# mod <- model.matrix(~as.factor(Relapse), data=ptr)
# mod0 <- model.matrix(~1, data=ptr)
# numsv <- num.sv(exprs, mod, method="be")
# svaobj <- sva(exprs, mod, mod0, method="supervised", n.sv=numsv, controls=controls)
# exprs_sva <- getSvaBcExprs(exprs, mod, svaobj)
# exprs(eset_gex_merged_sva) <- exprs_sva
