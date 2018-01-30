#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
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
    mod <- model.matrix(~as.factor(Relapse), data=ptr)
    mod0 <- model.matrix(~1, data=ptr)
    ctrls <- grepl("^AFFX", rownames(Xtr))
    n.sv <- sva::num.sv(Xtr, mod, method="be")
    ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
    eset_tr_ssva <- get(eset_tr_strs[i])
    exprs(eset_tr_ssva) <- t(ssva.params$xadj)
    eset_tr_ssva_str <- paste0(eset_tr_strs[i], "_tr_ssva")
    assign(eset_tr_ssva_str, eset_tr_ssva)
    save(list=eset_tr_ssva_str, file=paste0("data/", eset_tr_ssva_str, ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    Xte <- exprs(get(eset_te_strs[i]))
    eset_te_ssva <- get(eset_te_strs[i])
    exprs(eset_te_ssva) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
    eset_te_ssva_str <- paste0(eset_te_strs[i], "_te_ssva")
    assign(eset_te_ssva_str, eset_te_ssva)
    save(list=eset_te_ssva_str, file=paste0("data/", eset_te_ssva_str, ".Rda"))
    # sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
    # eset_tr_sva <- get(eset_tr_strs[i])
    # exprs(eset_tr_sva) <- t(sva.params$xadj)
    # eset_tr_sva_str <- paste0(eset_tr_strs[i], "_tr_sva")
    # assign(eset_tr_sva_str, eset_tr_sva)
    # save(list=eset_tr_sva_str, file=paste0("data/", eset_tr_sva_str, ".Rda"))
    # eset_te_sva <- get(eset_te_strs[i])
    # exprs(eset_te_sva) <- t(bapred::svabaaddon(sva.params, t(Xte)))
    # eset_te_sva_str <- paste0(eset_te_strs[i], "_te_sva")
    # assign(eset_te_sva_str, eset_te_sva)
    # save(list=eset_te_sva_str, file=paste0("data/", eset_te_sva_str, ".Rda"))
}

# regress surrogate vars out of exprs to get batch corrected exprs
# getSvaBcExprs <- function(exprs, mod, svaobj) {
#     X <- cbind(mod, svaobj$sv)
#     Hat <- solve(t(X) %*% X) %*% t(X)
#     beta <- (Hat %*% t(exprs))
#     P <- ncol(mod)
#     exprs_sva <- exprs - t(as.matrix(X[,-c(1:P)]) %*% beta[-c(1:P),])
#     return(exprs_sva)
# }

# mod <- model.matrix(~as.factor(Relapse), data=ptr)
# mod0 <- model.matrix(~1, data=ptr)
# n.sv <- num.sv(exprs, mod, method="be")
# svaobj <- sva(exprs, mod, mod0, method="supervised", n.sv=n.sv, controls=controls)
# exprs_sva <- getSvaBcExprs(exprs, mod, svaobj)
# exprs(eset_gex_merged_sva) <- exprs_sva
