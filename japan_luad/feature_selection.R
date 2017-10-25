#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("limma"))
# load("eset.Rda")

selectExpressionFeatures <- function(eset, relapse.fs.percent = .15, p.value = 0.05, lfc = 1.5, max.num.features = 50) {
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
    table <- topTable(fit.b, number=max.num.features, adjust.method="BH", p.value=p.value, lfc=lfc, sort.by="logFC")
    return(table)
}
