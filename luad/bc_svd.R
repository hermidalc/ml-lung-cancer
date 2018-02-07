#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")
source("config.R")

for (i in 1:length(eset_merged_tr_names)) {
    eset_tr_svd_name <- paste0(eset_merged_tr_names[i], "_tr_svd")
    eset_te_svd_name <- paste0(eset_merged_te_names[i], "_te_svd")
    print(paste(eset_tr_svd_name, "->", eset_te_svd_name))
    load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
    ptr <- pData(get(eset_merged_tr_names[i]))
    Xtr <- exprs(get(eset_merged_tr_names[i]))
    svd_obj <- normFact("SVD", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical")
    eset_tr_svd <- get(eset_merged_tr_names[i])
    exprs(eset_tr_svd) <- svd_obj$Xn
    assign(eset_tr_svd_name, eset_tr_svd)
    save(list=eset_tr_svd_name, file=paste0("data/", eset_tr_svd_name, ".Rda"))
    eset_tr_svd_obj_name <- paste0(eset_tr_svd_name, "_obj")
    assign(eset_tr_svd_obj_name, svd_obj)
    save(list=eset_tr_svd_obj_name, file=paste0("data/", eset_tr_svd_obj_name, ".Rda"))
    load(paste0("data/", eset_merged_te_names[i], ".Rda"))
    Xte <- exprs(get(eset_merged_te_names[i]))
    eset_te_svd <- get(eset_merged_te_names[i])
    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
    # Xte_n = dot(U,Vte.T)
    exprs(eset_te_svd) <- svd_obj$U %*% t((t(Xte) %*% svd_obj$U) %*% solve(t(svd_obj$U) %*% svd_obj$U))
    assign(eset_te_svd_name, eset_te_svd)
    save(list=eset_te_svd_name, file=paste0("data/", eset_te_svd_name, ".Rda"))
}
