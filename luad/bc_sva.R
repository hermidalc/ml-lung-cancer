#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")

load("data/eset_gex_gse31210_gse8894_gse30219_gse37745.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse30219_gse37745)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse30219_gse37745)
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
n.sv <- num.sv(Xtr, mod, method="be")
ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
eset_gex_gse31210_gse8894_gse30219_gse37745_ssva_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
eset_gex_gse31210_gse8894_gse30219_gse37745_sva_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_ssva_tr) <- t(ssva.params$xadj)
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_sva_tr) <- t(sva.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse37745_ssva_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_ssva_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse37745_sva_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_sva_tr.Rda")
load("data/eset_gex_gse50081.Rda")
Xte <- exprs(eset_gex_gse50081)
eset_gex_gse50081_ssva_te <- eset_gex_gse50081
eset_gex_gse50081_sva_te <- eset_gex_gse50081
exprs(eset_gex_gse50081_ssva_te) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
exprs(eset_gex_gse50081_sva_te) <- t(bapred::svabaaddon(sva.params, t(Xte)))
save(eset_gex_gse50081_ssva_te, file="data/eset_gex_gse50081_ssva_te.Rda")
save(eset_gex_gse50081_sva_te, file="data/eset_gex_gse50081_sva_te.Rda")
#
load("data/eset_gex_gse31210_gse8894_gse30219_gse50081.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse30219_gse50081)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse30219_gse50081)
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
n.sv <- num.sv(Xtr, mod, method="be")
ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
eset_gex_gse31210_gse8894_gse30219_gse50081_ssva_tr <- eset_gex_gse31210_gse8894_gse30219_gse50081
eset_gex_gse31210_gse8894_gse30219_gse50081_sva_tr <- eset_gex_gse31210_gse8894_gse30219_gse50081
exprs(eset_gex_gse31210_gse8894_gse30219_gse50081_ssva_tr) <- t(ssva.params$xadj)
exprs(eset_gex_gse31210_gse8894_gse30219_gse50081_sva_tr) <- t(sva.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse50081_ssva_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse50081_ssva_tr.Rda")
save(eset_gex_gse31210_gse8894_gse30219_gse50081_sva_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse50081_sva_tr.Rda")
load("data/eset_gex_gse37745.Rda")
Xte <- exprs(eset_gex_gse37745)
eset_gex_gse37745_ssva_te <- eset_gex_gse37745
eset_gex_gse37745_sva_te <- eset_gex_gse37745
exprs(eset_gex_gse37745_ssva_te) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
exprs(eset_gex_gse37745_sva_te) <- t(bapred::svabaaddon(sva.params, t(Xte)))
save(eset_gex_gse37745_ssva_te, file="data/eset_gex_gse37745_ssva_te.Rda")
save(eset_gex_gse37745_sva_te, file="data/eset_gex_gse37745_sva_te.Rda")
#
load("data/eset_gex_gse31210_gse8894_gse37745_gse50081.Rda")
ptr <- pData(eset_gex_gse31210_gse8894_gse37745_gse50081)
Xtr <- exprs(eset_gex_gse31210_gse8894_gse37745_gse50081)
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
n.sv <- num.sv(Xtr, mod, method="be")
ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
eset_gex_gse31210_gse8894_gse37745_gse50081_ssva_tr <- eset_gex_gse31210_gse8894_gse37745_gse50081
eset_gex_gse31210_gse8894_gse37745_gse50081_sva_tr <- eset_gex_gse31210_gse8894_gse37745_gse50081
exprs(eset_gex_gse31210_gse8894_gse37745_gse50081_ssva_tr) <- t(ssva.params$xadj)
exprs(eset_gex_gse31210_gse8894_gse37745_gse50081_sva_tr) <- t(sva.params$xadj)
save(eset_gex_gse31210_gse8894_gse37745_gse50081_ssva_tr, file="data/eset_gex_gse31210_gse8894_gse37745_gse50081_ssva_tr.Rda")
save(eset_gex_gse31210_gse8894_gse37745_gse50081_sva_tr, file="data/eset_gex_gse31210_gse8894_gse37745_gse50081_sva_tr.Rda")
load("data/eset_gex_gse30219.Rda")
Xte <- exprs(eset_gex_gse30219)
eset_gex_gse30219_ssva_te <- eset_gex_gse30219
eset_gex_gse30219_sva_te <- eset_gex_gse30219
exprs(eset_gex_gse30219_ssva_te) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
exprs(eset_gex_gse30219_sva_te) <- t(bapred::svabaaddon(sva.params, t(Xte)))
save(eset_gex_gse30219_ssva_te, file="data/eset_gex_gse30219_ssva_te.Rda")
save(eset_gex_gse30219_sva_te, file="data/eset_gex_gse30219_sva_te.Rda")
#
load("data/eset_gex_gse31210_gse30219_gse37745_gse50081.Rda")
ptr <- pData(eset_gex_gse31210_gse30219_gse37745_gse50081)
Xtr <- exprs(eset_gex_gse31210_gse30219_gse37745_gse50081)
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
n.sv <- num.sv(Xtr, mod, method="be")
ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
eset_gex_gse31210_gse30219_gse37745_gse50081_ssva_tr <- eset_gex_gse31210_gse30219_gse37745_gse50081
eset_gex_gse31210_gse30219_gse37745_gse50081_sva_tr <- eset_gex_gse31210_gse30219_gse37745_gse50081
exprs(eset_gex_gse31210_gse30219_gse37745_gse50081_ssva_tr) <- t(ssva.params$xadj)
exprs(eset_gex_gse31210_gse30219_gse37745_gse50081_sva_tr) <- t(sva.params$xadj)
save(eset_gex_gse31210_gse30219_gse37745_gse50081_ssva_tr, file="data/eset_gex_gse31210_gse30219_gse37745_gse50081_ssva_tr.Rda")
save(eset_gex_gse31210_gse30219_gse37745_gse50081_sva_tr, file="data/eset_gex_gse31210_gse30219_gse37745_gse50081_sva_tr.Rda")
load("data/eset_gex_gse8894.Rda")
Xte <- exprs(eset_gex_gse8894)
eset_gex_gse8894_ssva_te <- eset_gex_gse8894
eset_gex_gse8894_sva_te <- eset_gex_gse8894
exprs(eset_gex_gse8894_ssva_te) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
exprs(eset_gex_gse8894_sva_te) <- t(bapred::svabaaddon(sva.params, t(Xte)))
save(eset_gex_gse8894_ssva_te, file="data/eset_gex_gse8894_ssva_te.Rda")
save(eset_gex_gse8894_sva_te, file="data/eset_gex_gse8894_sva_te.Rda")
#
load("data/eset_gex_gse8894_gse30219_gse37745_gse50081.Rda")
ptr <- pData(eset_gex_gse8894_gse30219_gse37745_gse50081)
Xtr <- exprs(eset_gex_gse8894_gse30219_gse37745_gse50081)
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
n.sv <- num.sv(Xtr, mod, method="be")
ssva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, controls=ctrls, algorithm="exact")
sva.params <- svaba(t(Xtr), ytr, btr, mod, mod0, n.sv, algorithm="exact")
eset_gex_gse8894_gse30219_gse37745_gse50081_ssva_tr <- eset_gex_gse8894_gse30219_gse37745_gse50081
eset_gex_gse8894_gse30219_gse37745_gse50081_sva_tr <- eset_gex_gse8894_gse30219_gse37745_gse50081
exprs(eset_gex_gse8894_gse30219_gse37745_gse50081_ssva_tr) <- t(ssva.params$xadj)
exprs(eset_gex_gse8894_gse30219_gse37745_gse50081_sva_tr) <- t(sva.params$xadj)
save(eset_gex_gse8894_gse30219_gse37745_gse50081_ssva_tr, file="data/eset_gex_gse8894_gse30219_gse37745_gse50081_ssva_tr.Rda")
save(eset_gex_gse8894_gse30219_gse37745_gse50081_sva_tr, file="data/eset_gex_gse8894_gse30219_gse37745_gse50081_sva_tr.Rda")
load("data/eset_gex_gse31210.Rda")
Xte <- exprs(eset_gex_gse31210)
eset_gex_gse31210_ssva_te <- eset_gex_gse31210
eset_gex_gse31210_sva_te <- eset_gex_gse31210
exprs(eset_gex_gse31210_ssva_te) <- t(bapred::svabaaddon(ssva.params, t(Xte)))
exprs(eset_gex_gse31210_sva_te) <- t(bapred::svabaaddon(sva.params, t(Xte)))
save(eset_gex_gse31210_ssva_te, file="data/eset_gex_gse31210_ssva_te.Rda")
save(eset_gex_gse31210_sva_te, file="data/eset_gex_gse31210_sva_te.Rda")

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
