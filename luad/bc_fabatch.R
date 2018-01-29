#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))

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
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr <- eset_gex_gse31210_gse8894_gse30219_gse37745
exprs(eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse37745_fab_tr.Rda")
load("data/eset_gex_gse50081.Rda")
pte <- pData(eset_gex_gse50081)
Xte <- exprs(eset_gex_gse50081)
bte <- pte$Batch
bute <- sort(unique(bte))
for (i in 1:length(bute)) {
    if (i != bute[i]) {
        bte <- replace(bte, bte == bute[i], i)
    }
}
bte <- as.factor(bte)
eset_gex_gse50081_fab_te <- eset_gex_gse50081
exprs(eset_gex_gse50081_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse50081_fab_te, file="data/eset_gex_gse50081_fab_te.Rda")
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
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse31210_gse8894_gse30219_gse50081_fab_tr <- eset_gex_gse31210_gse8894_gse30219_gse50081
exprs(eset_gex_gse31210_gse8894_gse30219_gse50081_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse31210_gse8894_gse30219_gse50081_fab_tr, file="data/eset_gex_gse31210_gse8894_gse30219_gse50081_fab_tr.Rda")
load("data/eset_gex_gse37745.Rda")
pte <- pData(eset_gex_gse37745)
Xte <- exprs(eset_gex_gse37745)
bte <- pte$Batch
bute <- sort(unique(bte))
for (i in 1:length(bute)) {
    if (i != bute[i]) {
        bte <- replace(bte, bte == bute[i], i)
    }
}
bte <- as.factor(bte)
eset_gex_gse37745_fab_te <- eset_gex_gse37745
exprs(eset_gex_gse37745_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse37745_fab_te, file="data/eset_gex_gse37745_fab_te.Rda")
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
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse31210_gse8894_gse37745_gse50081_fab_tr <- eset_gex_gse31210_gse8894_gse37745_gse50081
exprs(eset_gex_gse31210_gse8894_gse37745_gse50081_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse31210_gse8894_gse37745_gse50081_fab_tr, file="data/eset_gex_gse31210_gse8894_gse37745_gse50081_fab_tr.Rda")
load("data/eset_gex_gse30219.Rda")
pte <- pData(eset_gex_gse30219)
Xte <- exprs(eset_gex_gse30219)
bte <- pte$Batch
bute <- sort(unique(bte))
for (i in 1:length(bute)) {
    if (i != bute[i]) {
        bte <- replace(bte, bte == bute[i], i)
    }
}
bte <- as.factor(bte)
eset_gex_gse30219_fab_te <- eset_gex_gse30219
exprs(eset_gex_gse30219_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse30219_fab_te, file="data/eset_gex_gse30219_fab_te.Rda")
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
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse31210_gse30219_gse37745_gse50081_fab_tr <- eset_gex_gse31210_gse30219_gse37745_gse50081
exprs(eset_gex_gse31210_gse30219_gse37745_gse50081_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse31210_gse30219_gse37745_gse50081_fab_tr, file="data/eset_gex_gse31210_gse30219_gse37745_gse50081_fab_tr.Rda")
load("data/eset_gex_gse8894.Rda")
pte <- pData(eset_gex_gse8894)
Xte <- exprs(eset_gex_gse8894)
bte <- pte$Batch
bute <- sort(unique(bte))
for (i in 1:length(bute)) {
    if (i != bute[i]) {
        bte <- replace(bte, bte == bute[i], i)
    }
}
bte <- as.factor(bte)
eset_gex_gse8894_fab_te <- eset_gex_gse8894
exprs(eset_gex_gse8894_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse8894_fab_te, file="data/eset_gex_gse8894_fab_te.Rda")
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
fab.params <- fabatch(t(Xtr), ytr, btr)
eset_gex_gse8894_gse30219_gse37745_gse50081_fab_tr <- eset_gex_gse8894_gse30219_gse37745_gse50081
exprs(eset_gex_gse8894_gse30219_gse37745_gse50081_fab_tr) <- t(fab.params$xadj)
save(eset_gex_gse8894_gse30219_gse37745_gse50081_fab_tr, file="data/eset_gex_gse8894_gse30219_gse37745_gse50081_fab_tr.Rda")
load("data/eset_gex_gse31210.Rda")
pte <- pData(eset_gex_gse31210)
Xte <- exprs(eset_gex_gse31210)
bte <- pte$Batch
bute <- sort(unique(bte))
for (i in 1:length(bute)) {
    if (i != bute[i]) {
        bte <- replace(bte, bte == bute[i], i)
    }
}
bte <- as.factor(bte)
eset_gex_gse31210_fab_te <- eset_gex_gse31210
exprs(eset_gex_gse31210_fab_te) <- t(fabatchaddon(fab.params, t(Xte), bte))
save(eset_gex_gse31210_fab_te, file="data/eset_gex_gse31210_fab_te.Rda")
