#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    eset_tr_svd_str <- paste0(eset_tr_strs[i], "_tr_svd")
    eset_te_svd_str <- paste0(eset_te_strs[i], "_te_svd")
    print(paste(eset_tr_svd_str, "->", eset_te_svd_str))
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    svd_obj <- normFact("SVD", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical")
    eset_tr_svd <- get(eset_tr_strs[i])
    exprs(eset_tr_svd) <- svd_obj$Xn
    assign(eset_tr_svd_str, eset_tr_svd)
    save(list=eset_tr_svd_str, file=paste0("data/", eset_tr_svd_str, ".Rda"))
    eset_tr_svd_obj_str <- paste0(eset_tr_svd_str, "_obj")
    assign(eset_tr_svd_obj_str, svd_obj)
    save(list=eset_tr_svd_obj_str, file=paste0("data/", eset_tr_svd_obj_str, ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    Xte <- exprs(get(eset_te_strs[i]))
    eset_te_svd <- get(eset_te_strs[i])
    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
    # Xte_n = dot(U,Vte.T)
    exprs(eset_te_svd) <- svd_obj$U %*% t((t(Xte) %*% svd_obj$U) %*% solve(t(svd_obj$U) %*% svd_obj$U))
    assign(eset_te_svd_str, eset_te_svd)
    save(list=eset_te_svd_str, file=paste0("data/", eset_te_svd_str, ".Rda"))
}
