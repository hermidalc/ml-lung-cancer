#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))
load("eset.Rda")

selectGeneExpressionFeatures <- function() {
    eset.fs <- eset[,sort(c(sample(which(eset$Relapse == 1), 10),sample(which(eset$Relapse == 0), 108)))]
    design <- model.matrix(~0 + factor(pData(eset.fs)$Relapse))
    colnames(design) <- c("NoRelapse", "Relapse")
    fit <- lmFit(eset.fs, design)
    contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    table <- topTable(fit.b)
    # results <- decideTests(fit.b)
    # summary(results)
    
    return(table)
}
