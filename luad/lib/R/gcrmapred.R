gcrmatrain <- function(affybatchtrain) {
    # perform GCRMA
    abg <- gcrma::bg.adjust.gcrma(affybatchtrain, type="fullmodel", verbose=TRUE, fast=FALSE)
    a.nrm.rma <- bapred::normalizeAffyBatchqntval(abg,'pmonly')
    # store parameters for add-on quantile normalization
    rmadoc <- Biobase::experimentData(a.nrm.rma)@preprocessing[['val']]
    summ.rma <- bapred::summarizeval2(a.nrm.rma)
    sumdoc.rma <- Biobase::experimentData(summ.rma)@preprocessing$val$probe.effects
    # extract gene expressions
    exprs.train.rma <- exprs(summ.rma)
    gcrma_obj <- list(xnorm=t(exprs.train.rma), rmadoc=rmadoc, sumdoc.rma=sumdoc.rma, nfeature=nrow(exprs.train.rma))
    class(gcrma_obj) <- "gcrmatrain"
    return(gcrma_obj)
}

gcrmaaddon <- function(gcrma_obj, affybatchtest) {
    if (class(gcrma_obj) != "gcrmatrain")
        stop("Input parameter 'gcrma_obj' has to be of class 'gcrmatrain'.")
    # perform GCRMA with add-on quantile normalization
    exprs.test.rma <- matrix(0,nrow=length(affybatchtest),ncol=gcrma_obj$nfeature)
    for (cel in 1:length(affybatchtest)) {
        system.time(ab.add <- bapred::extractAffybatch(cel, affybatchtest))
        abg <- gcrma::bg.adjust.gcrma(ab.add, type="fullmodel", verbose=TRUE, fast=FALSE)
        abo.nrm.rma  <- bapred::normalizeqntadd(abg, gcrma_obj$rmadoc$mqnts)
        eset <- bapred::summarizeadd2(abo.nrm.rma, gcrma_obj$sumdoc.rma)
        exprs.test.rma[cel,] <- t(exprs(eset))
    }
    return(exprs.test.rma)
}
