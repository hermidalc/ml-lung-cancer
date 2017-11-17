#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))

randPermSampleNums <- function(eset, is.relapse) {
    return(sample(which(eset$Relapse == is.relapse)))
}

filterEsetSamples <- function(eset, relapse.samples, norelapse.samples) {
    return(eset[,sort(c(relapse.samples, norelapse.samples))])
}

filterEsetFeatures <- function(eset, features) {
    return (eset[sort(c(features)),])
}

# selectExpFeatures <- function(eset, min.p.value = 0.05, min.lfc = 1.5, max.num.features = 50) {
selectExpFeatures <- function(eset, max.num.features = 50) {
    design <- model.matrix(~0 + factor(pData(eset)$Relapse))
    colnames(design) <- c("NoRelapse", "Relapse")
    fit <- lmFit(eset, design)
    contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    # results <- decideTests(fit.b, method="global")
    # summary(results)
    # return(topTable(fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="logFC"))
    return(topTable(fit.b, number=max.num.features, adjust.method="BH", sort.by="logFC"))
}
