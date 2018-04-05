gcrmatrain <- function(affybatchtrain, affinities) {
    # perform GCRMA
    abg <- gcrma::bg.adjust.gcrma(affybatchtrain, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
    a.nrm.rma <- bapred::normalizeAffyBatchqntval(abg, 'pmonly')
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

gcrmaaddon <- function(gcrma_obj, affybatchtest, affinities, parallel=TRUE) {
    if (class(gcrma_obj) != "gcrmatrain")
        stop("Input parameter 'gcrma_obj' has to be of class 'gcrmatrain'.")
    # perform GCRMA with add-on quantile normalization
    abg <- gcrma::bg.adjust.gcrma(affybatchtest, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
    cat("Performing add-on normalization/summarization")
    if (parallel) {
        suppressPackageStartupMessages(require("doParallel"))
        registerDoParallel(cores=max(detectCores()/2, 1))
        exprs.test.rma <- foreach (cel=1:length(abg), .combine="cbind") %dopar% {
            ab.add <- bapred::extractAffybatch(cel, abg)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, gcrma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, gcrma_obj$sumdoc.rma)
            cat(".")
            exprs(eset)
        }
    }
    else {
        exprs.test.rma <- matrix(0, nrow=gcrma_obj$nfeature, ncol=length(abg))
        for (cel in 1:length(abg)) {
            ab.add <- bapred::extractAffybatch(cel, abg)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, gcrma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, gcrma_obj$sumdoc.rma)
            exprs.test.rma[,cel] <- exprs(eset)
            cat(".")
        }
    }
    cat("Done.\n")
    return(t(exprs.test.rma))
}
