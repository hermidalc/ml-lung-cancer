suppressPackageStartupMessages(library("Biobase"))
set.seed(1982)

randPermSampleNums <- function(eset, is.positive) {
    is.positive.num <- if (is.positive) 1 else 0
    return(sample(which(eset$Class == is.positive.num)))
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
    suppressPackageStartupMessages(require("genefilter"))
    return(featureFilter(eset,
        require.entrez=FALSE,
        require.GOBP=FALSE, require.GOCC=FALSE,
        require.GOMF=FALSE, require.CytoBand=FALSE,
        remove.dupEntrez=FALSE, feature.exclude="^AFFX"
    ))
}

getEsetClassLabels <- function(eset, samples=NULL) {
    if (!is.null(samples)) {
        return(eset$Class[c(samples)])
    }
    else {
        return(eset$Class)
    }
}

getEsetGeneSymbols <- function(eset, features=NULL) {
    if (!is.null(features)) {
        symbols <- as.character(featureData(eset)[c(features)]$Symbol)
    }
    else {
        symbols <- as.character(featureData(eset)$Symbol)
    }
    symbols[is.na(symbols)] <- ""
    return(symbols)
}

limmaFeatureScore <- function(exprs, class) {
    suppressPackageStartupMessages(require("limma"))
    design <- model.matrix(~0 + factor(class))
    colnames(design) <- c("Class0", "Class1")
    fit <- lmFit(exprs, design)
    contrast.matrix <- makeContrasts(Class1VsClass0=Class1-Class0, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    results <- topTableF(fit.b, number=Inf, adjust.method="BH")
    results <- results[order(as.integer(row.names(results))),]
    return(list(results$F, results$adj.P.Val))
}

limmaFeatures <- function(eset, numbers=TRUE, min.p.value=0.05, min.lfc=0, max.num.features=Inf) {
    suppressPackageStartupMessages(require("limma"))
    design <- model.matrix(~0 + factor(pData(eset)$Class))
    colnames(design) <- c("Class0", "Class1")
    fit <- lmFit(eset, design)
    contrast.matrix <- makeContrasts(Class1VsClass0=Class1-Class0, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    feature.names <- rownames(topTable(
        fit.b, number=max.num.features, p.value=min.p.value, lfc=min.lfc, adjust.method="BH", sort.by="P"
    ))
    if (numbers) {
        return(which(featureNames(eset) %in% feature.names))
    }
    else {
        return(feature.names)
    }
}

fcbfFeatureIdxs <- function(X, y) {
    results <- Biocomb::select.fast.filter(cbind(X, as.factor(y)), disc.method="MDL", threshold=0)
    return(results$NumberFeature - 1)
}

cfsFeatureIdxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    feature_idxs <- FSelector::cfs(as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)))
    return(as.integer(feature_idxs) - 1)
}

gainRatioFeatureIdxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::gain.ratio(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(results$attr_importance, decreasing=TRUE), , drop=FALSE]
    return(as.integer(row.names(results)) - 1)
}

symmUncertFeatureIdxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::symmetrical.uncertainty(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(results$attr_importance, decreasing=TRUE), , drop=FALSE]
    return(as.integer(row.names(results)) - 1)
}

relieffFeatureIdxs <- function(X, y, num.neighbors=15, sample.size=10) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)),
        neighbours.count=num.neighbors, sample.size=sample.size
    )
    return(results[order(results$attr_importance, decreasing=TRUE), , drop=FALSE])
}
