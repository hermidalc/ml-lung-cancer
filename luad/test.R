#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))

selectExpFeatures <- function(eset, relapse.fs.percent=0.15, min.p.value=0.001, min.lfc=0, max.num.features=50) {
    num.relapse <- ncol(eset[,eset$Relapse == 1])
    # num.relapse.fs <- ceiling(ncol(eset[,eset$Relapse == 1]) * relapse.fs.percent)
    num.relapse.fs <- 100
    num.norelapse.fs <- ncol(eset[,eset$Relapse == 0]) - num.relapse + num.relapse.fs
    eset.fs <- eset[,sort(c(sample(which(eset$Relapse == 1), num.relapse.fs),sample(which(eset$Relapse == 0), num.norelapse.fs)))]
    design <- model.matrix(~0 + factor(pData(eset.fs)$Relapse))
    colnames(design) <- c("NoRelapse", "Relapse")
    fit <- lmFit(eset.fs, design)
    contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    # results <- decideTests(fit.b, method="global")
    # summary(results)
    # return(topTable(fit.b, number=max.num.features, adjust.method="BH", p.value=min.p.value, lfc=min.lfc, sort.by="logFC"))
    print(topTable(fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="P"))
    return(topTable(fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="P"))
}

source("functions.R")
load("data/eset_gex_gse31210_gse8894_gse30219_gse50081_stica05_tr.Rda")
features.df <- selectExpFeatures(eset_gex_gse31210_gse8894_gse30219_gse50081_stica05_tr)
# num.samples.tr <- num.relapse - (num.relapse.fs * 2)
# relapse.samples.tr <- relapse.samples[(num.relapse.fs + 1):(num.relapse.fs + num.samples.tr)]
# norelapse.samples.tr <- norelapse.samples[(num.norelapse.fs + 1):(num.norelapse.fs + num.samples.tr)]
# eset.tr <- filterEset(eset.gex, rownames(features.df), c(relapse.samples.tr, norelapse.samples.tr))
# data.tr <- t(exprs(eset.tr))
# labels.tr <- filterEsetRelapseLabels(eset.tr)
