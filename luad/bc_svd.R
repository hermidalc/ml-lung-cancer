#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    svdobj <- normFact("SVD", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical")
    eset_tr_svd <- get(eset_tr_strs[i])
    exprs(eset_tr_svd) <- svdobj$Xn
    eset_tr_svd_str <- paste0(eset_tr_strs[i], "_tr_svd")
    assign(eset_tr_svd_str, eset_tr_svd)
    save(list=eset_tr_svd_str, file=paste0("data/", eset_tr_svd_str, ".Rda"))
    save(list=svdobj, file=paste0("data/", eset_tr_svd_str, "_obj.Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    Xte <- exprs(get(eset_te_strs[i]))
    eset_te_svd <- get(eset_te_strs[i])
    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
    # Xte_n = dot(U,Vte.T)
    exprs(eset_te_svd) <- svdobj$U %*% t((t(Xte) %*% svdobj$U) %*% solve(t(svdobj$U) %*% svdobj$U))
    eset_te_svd_str <- paste0(eset_te_strs[i], "_te_svd")
    assign(eset_te_svd_str, eset_te_svd)
    save(list=eset_te_svd_str, file=paste0("data/", eset_te_svd_str, ".Rda"))
}
