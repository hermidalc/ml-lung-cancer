#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
source("normFact.R")

load("data/eset_gex_gse31210_gse8894_gse30219_gse37745.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse30219_gse37745)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse30219_gse37745)
stica0obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=0)
stica025obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=0.25)
stica05obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=0.5)
stica075obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=0.75)
stica1obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=1)
eset_gex_gse31210_gse8894_gse30219_gse37745_stica0_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
eset_gex_gse31210_gse8894_gse30219_gse37745_stica025_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
eset_gex_gse31210_gse8894_gse30219_gse37745_stica05_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
eset_gex_gse31210_gse8894_gse30219_gse37745_stica075_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
eset_gex_gse31210_gse8894_gse30219_gse37745_stica1_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_stica0_tr) <- stica0obj$Xn
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_stica025_tr) <- stica025obj$Xn
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_stica05_tr) <- stica05obj$Xn
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_stica075_tr) <- stica075obj$Xn
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_stica1_tr) <- stica1obj$Xn
save(eset_gex_gse31210_gse8894_gse30219_gse37745_stica0_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_stica0_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse37745_stica025_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_stica025_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse37745_stica05_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_stica05_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse37745_stica075_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_stica075_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse37745_stica1_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_stica1_tr.Rda")
load("data/eset_gex_gse50081.Rda")
Xte <- exprs(eset_gex_gse50081)
eset_gex_gse50081_stica0_te <- eset_gex_gse50081
eset_gex_gse50081_stica025_te <- eset_gex_gse50081
eset_gex_gse50081_stica05_te <- eset_gex_gse50081
eset_gex_gse50081_stica075_te <- eset_gex_gse50081
eset_gex_gse50081_stica1_te <- eset_gex_gse50081
# from stICA paper code addon batch effect correction
# Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
# Xte_n = dot(U,Vte.T)
exprs(eset_gex_gse50081_stica0_te) <- stica0obj$U %*% t((t(Xte) %*% stica0obj$U) %*% solve(t(stica0obj$U) %*% stica0obj$U))
exprs(eset_gex_gse50081_stica025_te) <- stica025obj$U %*% t((t(Xte) %*% stica025obj$U) %*% solve(t(stica025obj$U) %*% stica025obj$U))
exprs(eset_gex_gse50081_stica05_te) <- stica05obj$U %*% t((t(Xte) %*% stica05obj$U) %*% solve(t(stica05obj$U) %*% stica05obj$U))
exprs(eset_gex_gse50081_stica075_te) <- stica075obj$U %*% t((t(Xte) %*% stica075obj$U) %*% solve(t(stica075obj$U) %*% stica075obj$U))
exprs(eset_gex_gse50081_stica0_te) <- stica1obj$U %*% t((t(Xte) %*% stica1obj$U) %*% solve(t(stica1obj$U) %*% stica1obj$U))
save(eset_gex_gse50081_stica0_te, file="data/eset_gex_gse50081_stica0_te.Rda")
save(eset_gex_gse50081_stica025_te, file="data/eset_gex_gse50081_stica025_te.Rda")
save(eset_gex_gse50081_stica05_te, file="data/eset_gex_gse50081_stica05_te.Rda")
save(eset_gex_gse50081_stica075_te, file="data/eset_gex_gse50081_stica075_te.Rda")
save(eset_gex_gse50081_stica1_te, file="data/eset_gex_gse50081_stica1_te.Rda")
