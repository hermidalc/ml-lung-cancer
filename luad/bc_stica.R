#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")
source("config.R")

for (i in 1:length(eset_tr_strs)) {
    load(paste0("data/", eset_tr_strs[i], ".Rda"))
    load(paste0("data/", eset_te_strs[i], ".Rda"))
    ptr <- pData(get(eset_tr_strs[i]))
    Xtr <- exprs(get(eset_tr_strs[i]))
    Xte <- exprs(get(eset_te_strs[i]))
    for (alpha in stica_alphas) {
        alpha_str <- gsub("[^0-9]", "", alpha)
        sticaobj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=alpha)
        eset_tr_stica <- get(eset_tr_strs[i])
        exprs(eset_tr_stica) <- sticaobj$Xn
        eset_tr_stica_str <- paste0(eset_tr_strs[i], "_tr_stica", alpha_str)
        assign(eset_tr_stica_str, eset_tr_stica)
        save(get(eset_tr_stica_str), file=paste0("data/", eset_tr_stica_str, ".Rda"))
        eset_te_stica <- get(eset_te_strs[i])
        # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
        # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
        # Xte_n = dot(U,Vte.T)
        exprs(eset_te_stica) <- sticaobj$U %*% t((t(Xte) %*% sticaobj$U) %*% solve(t(sticaobj$U) %*% sticaobj$U))
        eset_te_stica_str <- paste0(eset_te_strs[i], "_te_stica", alpha_str)
        assign(eset_te_stica_str, eset_te_stica)
        save(get(eset_te_stica_str), file=paste0("data/", eset_te_stica_str, ".Rda"))
    }
}
