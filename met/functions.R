suppressPackageStartupMessages(library("Biobase"))
set.seed(1982)

datasetX <- function(X_df, p_df, batch=NULL) {
    if (!is.null(batch)) {
        return(as.matrix(X_df[!is.na(p_df$Class) & p_df$Batch %in% c(batch),]))
    } else {
        return(as.matrix(X_df[!is.na(p_df$Class),]))
    }
}

datasetY <- function(p_df, batch=NULL) {
    if (!is.null(batch)) {
        return(as.integer(p_df[!is.na(p_df$Class) & p_df$Batch %in% c(batch), "Class"]))
    } else {
        return(as.integer(p_df[!is.na(p_df$Class), "Class"]))
    }
}

datasetNonZeroStdIdxs <- function(X, samples=FALSE) {
    if (samples) {
        return(as.integer(which(sapply(as.data.frame(t(X)), function(c) sd(c) != 0))) - 1)
    } else {
        return(as.integer(which(sapply(as.data.frame(X), function(c) sd(c) != 0))) - 1)
    }
}

datasetCorrIdxs <- function(X, cutoff=0.5, samples=FALSE) {
    if (samples) {
        return(sort(caret::findCorrelation(cor(t(X)), cutoff=cutoff)) - 1)
    } else {
        return(sort(caret::findCorrelation(cor(X), cutoff=cutoff)) - 1)
    }
}

datasetIdxs <- function(X_df, p_df, batch=NULL) {
    if (!is.null(batch)) {
        return(which(!is.na(p_df$Class) & p_df$Batch %in% c(batch)) - 1)
    } else {
        return(which(!is.na(p_df$Class)) - 1)
    }
}

limmaFeatureScore <- function(X, y) {
    suppressPackageStartupMessages(require("limma"))
    design <- model.matrix(~0 + factor(y))
    colnames(design) <- c("Class0", "Class1")
    fit <- lmFit(t(X), design)
    contrast.matrix <- makeContrasts(Class1VsClass0=Class1-Class0, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    results <- topTableF(fit.b, number=Inf, adjust.method="BH")
    results <- results[order(as.integer(row.names(results))),]
    return(list(results$F, results$adj.P.Val))
}

limmaFpkmFeatureScore <- function(X, y) {
    suppressPackageStartupMessages(require("limma"))
    design <- model.matrix(~0 + factor(y))
    colnames(design) <- c("Class0", "Class1")
    fit <- lmFit(t(log2(X + 1)), design)
    contrast.matrix <- makeContrasts(Class1VsClass0=Class1-Class0, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts, trend=TRUE)
    results <- topTableF(fit.b, number=Inf, adjust.method="BH")
    results <- results[order(as.integer(row.names(results))),]
    return(list(results$F, results$adj.P.Val))
}

fcbfFeatureIdxs <- function(X, y, threshold=0) {
    results <- Biocomb::select.fast.filter(cbind(X, as.factor(y)), disc.method="MDL", threshold=threshold)
    results <- results[order(results$Information.Gain, decreasing=TRUE), , drop=FALSE]
    return(results$NumberFeature - 1)
}

cfsFeatureIdxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    feature_idxs <- FSelector::cfs(as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)))
    return(as.integer(feature_idxs) - 1)
}

relieffFeatureScore <- function(X, y, num.neighbors=10, sample.size=5) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)),
        neighbours.count=num.neighbors, sample.size=sample.size
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(results$attr_importance)
}