#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("genefilter"))
suppressPackageStartupMessages(library("limma"))
# set.seed(1982)

randPermSampleNums <- function(eset, is.relapse) {
    is.relapse.num <- if (is.relapse) 1 else 0
    return(sample(which(eset$Relapse == is.relapse.num)))
}

filterEset <- function(eset, features=NULL, samples=NULL) {
    if (!is.null(features) && !is.null(samples)) {
        return(eset[c(features),c(samples)])
    }
    else if (!is.null(features)) {
        return(eset[c(features),])
    }
    else if (!is.null(samples)) {
        return(eset[,c(samples)])
    }
    else {
        return(eset)
    }
}

filterEsetControlProbesets <- function(eset) {
    return(featureFilter(eset,
        require.entrez=FALSE,
        require.GOBP=FALSE, require.GOCC=FALSE,
        require.GOMF=FALSE, require.CytoBand=FALSE,
        remove.dupEntrez=FALSE, feature.exclude="^AFFX"
    ))
}

filterEsetRelapseLabels <- function(eset, samples=NULL) {
    if (!is.null(samples)) {
        return(eset$Relapse[c(samples)])
    }
    else {
        return(eset$Relapse)
    }
}

getGeneSymbols <- function(eset, features=NULL) {
    if (!is.null(features)) {
        symbols <- as.character(featureData(eset)[c(features)]$Symbol)

    }
    else {
        symbols <- as.character(featureData(eset)$Symbol)
    }
    symbols[is.na(symbols)] <- ""
    return(symbols)
}

getDfxFeatures <- function(eset, numbers=FALSE, min.p.value=0.05, min.lfc=1, max.num.features=100) {
    design <- model.matrix(~0 + factor(pData(eset)$Relapse))
    colnames(design) <- c("NoRelapse", "Relapse")
    fit <- lmFit(eset, design)
    contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    feature.names <- rownames(topTable(fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="P"))
    if (numbers) {
        return(which(featureNames(eset) %in% feature.names))
    }
    else {
        return(feature.names)
    }
}
