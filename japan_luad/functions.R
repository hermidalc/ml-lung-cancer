#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))

randPermSampleNums <- function(eset, is.relapse) {
    is.relapse.num <- if (is.relapse) 1 else 0
    return(sample(which(eset$Relapse == is.relapse.num)))
}

filterEset <- function(eset, features=NULL, samples=NULL) {
    if (!is.null(features) && !is.null(samples)) {
        return (eset[c(features),c(samples)])
    }
    else if (!is.null(features)) {
        return (eset[c(features),])
    }
    else if (!is.null(samples)) {
        return (eset[,c(samples)])
    }
    else {
        return (eset)
    }
}

filterEsetRelapseLabels <- function(eset, samples=NULL) {
    if (!is.null(samples)) {
        return (eset$Relapse[c(samples)])
    }
    else {
        return (eset$Relapse)
    }
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
    # return(topTable(fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="logFC"))
    return(topTable(fit.b, number=max.num.features, adjust.method="BH", sort.by="logFC"))
}
