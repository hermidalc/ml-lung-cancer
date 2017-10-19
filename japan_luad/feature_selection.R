#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))
# load("eset.Rda")

selectGeneExpressionFeatures <- function(eset) {
    eset.fs <- eset[,sort(c(sample(which(eset$Relapse == 1), 10),sample(which(eset$Relapse == 0), 108)))]
    design <- model.matrix(~0 + factor(pData(eset.fs)$Relapse))
    colnames(design) <- c("NoRelapse", "Relapse")
    fit <- lmFit(eset.fs, design)
    contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
    fit.contrasts <- contrasts.fit(fit, contrast.matrix)
    fit.b <- eBayes(fit.contrasts)
    # results <- decideTests(fit.b, method="global")
    # summary(results)
    table <- topTable(fit.b, number=Inf, adjust.method="BH", p.value=0.01, lfc=1.5, sort.by="logFC")
    return(table)
}
