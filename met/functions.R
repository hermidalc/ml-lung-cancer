suppressPackageStartupMessages(library("Biobase"))

eset_class_labels <- function(eset, samples=NULL) {
    if (!is.null(samples)) {
        return(eset$Class[c(samples)])
    }
    else {
        return(eset$Class)
    }
}

eset_feature_idxs <- function(eset, features) {
    return(as.integer(which(rownames(eset) %in% features)) - 1)
}

eset_feature_annots <- function(eset, annots=annots, features=NULL) {
    if (!is.null(features)) {
        annots <- as.matrix(fData(eset)[c(features), c(annots), drop=FALSE])
    }
    else {
        annots <- as.matrix(fData(eset)[, c(annots), drop=FALSE])
    }
    annots[is.na(annots)] <- ""
    return(annots)
}

data_nzero_col_idxs <- function(X) {
    return(as.integer(which(colSums(X) > 0)) - 1)
}

data_nzero_sd_col_idxs <- function(X) {
    return(as.integer(which(sapply(as.data.frame(X), function(c) sd(c) != 0))) - 1)
}

data_nzero_var_col_idxs <- function(X, freqCut=95/5, uniqueCut=1) {
    return(sort(setdiff(1:ncol(X), caret::nearZeroVar(X, freqCut=freqCut, uniqueCut=uniqueCut))) - 1)
}

data_corr_col_idxs <- function(X, cutoff=0.5) {
    return(sort(caret::findCorrelation(cor(X), cutoff=cutoff)) - 1)
}

limma_feature_score <- function(X, y, pkm=FALSE) {
    suppressPackageStartupMessages(require("limma"))
    design <- model.matrix(~0 + factor(y))
    colnames(design) <- c("Class0", "Class1")
    if (pkm) {
        fit <- lmFit(t(log2(X + 1)), design)
    } else {
        fit <- lmFit(t(X), design)
    }
    contrast.matrix <- makeContrasts(Class1VsClass0=Class1-Class0, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    if (pkm) {
        fit.b <- eBayes(fit.contrasts, trend=TRUE)
    } else {
        fit.b <- eBayes(fit.contrasts)
    }
    results <- topTableF(fit.b, number=Inf, adjust.method="BH")
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(results$F, results$adj.P.Val))
}

fcbf_feature_idxs <- function(X, y, threshold=0) {
    results <- Biocomb::select.fast.filter(cbind(X, as.factor(y)), disc.method="MDL", threshold=threshold)
    results <- results[order(results$NumberFeature), , drop=FALSE]
    return(list(results$NumberFeature - 1, results$Information.Gain))
}

cfs_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    feature_idxs <- FSelector::cfs(as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)))
    return(as.integer(feature_idxs) - 1)
}

gain_ratio_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::gain.ratio(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

sym_uncert_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::symmetrical.uncertainty(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

relieff_feature_score <- function(X, y, num.neighbors=10, sample.size=5) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)),
        neighbours.count=num.neighbors, sample.size=sample.size
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(results$attr_importance)
}
