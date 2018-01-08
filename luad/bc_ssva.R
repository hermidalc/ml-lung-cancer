#!/usr/bin/env R

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("genefilter"))
suppressPackageStartupMessages(library("sva"))

# regress surrogate vars out of exprs to get batch corrected exprs
getSvaBcExprs <- function(exprs, mod, svaobj) {
    X <- cbind(mod, svaobj$sv)
    Hat <- solve(t(X) %*% X) %*% t(X)
    beta <- (Hat %*% t(exprs))
    P <- ncol(mod)
    exprs_sva <- exprs - t(as.matrix(X[,-c(1:P)]) %*% beta[-c(1:P),])
    return(exprs_sva)
}

load("data/eset_gex_merged.Rda")
pheno <- pData(eset_gex_merged)
exprs <- exprs(eset_gex_merged)
mod <- model.matrix(~as.factor(Relapse), data=pheno)
mod0 <- model.matrix(~1, data=pheno)
n.sv <- num.sv(exprs, mod, method="be")
controls <- grepl("^AFFX", rownames(exprs))
svaobj <- sva(exprs, mod, mod0, method="supervised", n.sv=n.sv, controls=controls)
exprs_sva <- getSvaBcExprs(exprs, mod, svaobj)
eset_gex_merged_sva <- eset_gex_merged
exprs(eset_gex_merged_sva) <- exprs_sva
# filter out control probesets
eset_gex_merged_sva <- featureFilter(eset_gex_merged_sva,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
eset_gex_gse31210_sva <- eset_gex_merged_sva[, eset_gex_merged_sva$Batch == 1]
eset_gex_gse30219_sva <- eset_gex_merged_sva[, eset_gex_merged_sva$Batch == 2]
eset_gex_gse37745_sva <- eset_gex_merged_sva[, eset_gex_merged_sva$Batch == 3]
eset_gex_gse50081_sva <- eset_gex_merged_sva[, eset_gex_merged_sva$Batch == 4]
save(eset_gex_merged_sva, file="data/eset_gex_merged_sva.Rda")
save(eset_gex_gse31210_sva, file="data/eset_gex_gse31210_sva.Rda")
save(eset_gex_gse30219_sva, file="data/eset_gex_gse30219_sva.Rda")
save(eset_gex_gse37745_sva, file="data/eset_gex_gse37745_sva.Rda")
save(eset_gex_gse50081_sva, file="data/eset_gex_gse50081_sva.Rda")
