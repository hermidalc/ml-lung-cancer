#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")
source("config.R")

for (i in 1:length(eset_merged_tr_names)) {
    load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
    load(paste0("data/", eset_merged_te_names[i], ".Rda"))
    ptr <- pData(get(eset_merged_tr_names[i]))
    Xtr <- exprs(get(eset_merged_tr_names[i]))
    Xte <- exprs(get(eset_merged_te_names[i]))
    for (alpha in stica_alphas) {
        alpha_name <- gsub("[^0-9]", "", alpha)
        eset_tr_stica_name <- paste0(eset_merged_tr_names[i], "_tr_stica", alpha_name)
        eset_te_stica_name <- paste0(eset_merged_te_names[i], "_te_stica", alpha_name)
        print(paste(eset_tr_stica_name, "->", eset_te_stica_name))
        stica_obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=alpha)
        eset_tr_stica <- get(eset_merged_tr_names[i])
        exprs(eset_tr_stica) <- stica_obj$Xn
        assign(eset_tr_stica_name, eset_tr_stica)
        save(list=eset_tr_stica_name, file=paste0("data/", eset_tr_stica_name, ".Rda"))
        eset_tr_stica_obj_name <- paste0(eset_tr_stica_name, "_obj")
        assign(eset_tr_stica_obj_name, stica_obj)
        save(list=eset_tr_stica_obj_name, file=paste0("data/", eset_tr_stica_obj_name, ".Rda"))
        eset_te_stica <- get(eset_merged_te_names[i])
        # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
        # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
        # Xte_n = dot(U,Vte.T)
        exprs(eset_te_stica) <- stica_obj$U %*% t((t(Xte) %*% stica_obj$U) %*% solve(t(stica_obj$U) %*% stica_obj$U))
        assign(eset_te_stica_name, eset_te_stica)
        save(list=eset_te_stica_name, file=paste0("data/", eset_te_stica_name, ".Rda"))
    }
}
