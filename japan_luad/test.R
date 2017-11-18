#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))

# selectExpFeatures <- function(eset, relapse.fs.percent = .15, min.p.value = 0.05, min.lfc = 1.5, max.num.features = 50) {
selectExpFeatures <- function(eset, relapse.fs.percent = .15, max.num.features = 50) {
    num.relapse <- ncol(eset[,eset$Relapse == 1])
    num.relapse.fs <- ceiling(ncol(eset[,eset$Relapse == 1]) * relapse.fs.percent)
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
    return(topTable(fit.b, number=max.num.features, adjust.method="BH", sort.by="logFC"))
}

source("functions.R")
load("eset_gex.Rda")
relapse.fs.percent <- .15
relapse.sample.nums <- randPermSampleNums(eset.gex, 1)
norelapse.sample.nums <- randPermSampleNums(eset.gex, 0)
num.relapse <- ncol(eset.gex[,eset.gex$Relapse == 1])
num.relapse.fs <- ceiling(ncol(eset.gex[,eset.gex$Relapse == 1]) * relapse.fs.percent)
num.norelapse.fs <- ncol(eset.gex[,eset.gex$Relapse == 0]) - num.relapse + num.relapse.fs
relapse.sample.nums.fs <- relapse.sample.nums[1:num.relapse.fs]
norelapse.sample.nums.fs <- norelapse.sample.nums[1:num.norelapse.fs]
eset.fs <- filterEset(eset.gex, NULL, c(norelapse.sample.nums.fs, relapse.sample.nums.fs))
features.df <- selectExpFeatures(eset.fs)
num.samples.tr <- num.relapse - (num.relapse.fs * 2)
relapse.sample.nums.tr <- relapse.sample.nums[(num.relapse.fs + 1):(num.relapse.fs + num.samples.tr)]
norelapse.sample.nums.tr <- norelapse.sample.nums[(num.norelapse.fs + 1):(num.norelapse.fs + num.samples.tr)]
eset.tr <- filterEset(eset.gex, rownames(features.df), c(relapse.sample.nums.tr, norelapse.sample.nums.tr))
data.tr <- exprs(eset.tr)
labels.tr <- filterEsetRelapseLabels(eset.gex, c(relapse.sample.nums.tr, norelapse.sample.nums.tr))
